#!/usr/bin/env python3
# grid_detection_v2.py

'''
############################################################
Version: 3.3
Key Changes: 
-after detecting columns, blurs (7,7) the crops to reduce noise
############################################################
'''




import os
import math
import cv2
import numpy as np
from ultralytics import YOLO
from math import sin, cos, radians


# --- Configurables (podés cambiarlos afuera si querés) ---
MODEL_PATH = "runs/segment/train20/weights/best.pt"
IMGSZ = 1024

CLASS_NAMES = ["col1", "col2", "col3"]

# Alias -> nombre canónico (ajustá según tu entrenamiento)
CLASS_ALIAS = {
    "cpl3": "col3",
    # "student_name": "colX",  # ejemplo si migrás nombres
}

# --- Helpers ---
def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # TL
    rect[2] = pts[np.argmax(s)]  # BR
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # TR
    rect[3] = pts[np.argmax(diff)]  # BL
    return rect

_model_cache = {"m": None}

def load_model():
    if _model_cache["m"] is None:
        _model_cache["m"] = YOLO(MODEL_PATH)
    return _model_cache["m"]

def _approx_quad_or_box(seg_pts: np.ndarray) -> np.ndarray:
    """
    Devuelve 4 puntos (float32) en orden arbitrario para warpear:
    - Intenta aproximación poligonal a 4 vértices.
    - Si falla, usa minAreaRect (caja rotada) para generar 4 puntos.
    """
    contour = seg_pts.reshape(-1, 1, 2).astype(np.float32)
    peri = cv2.arcLength(contour, True)
    eps = 0.02 * peri
    approx = cv2.approxPolyDP(contour, eps, True)
    if len(approx) == 4:
        return approx.reshape(4, 2).astype(np.float32)

    # Fallback robusto: caja mínima rotada
    r = cv2.minAreaRect(contour)
    box = cv2.boxPoints(r)  # 4x2
    return box.astype(np.float32)

def detect_grids(
    image_path: str,
    conf_threshold: float = 0.9,
    save: bool = False
) -> list[dict]:
    """
    Detecta y procesa regiones de grillas (vía YOLO-Seg).
    Devuelve la MISMA estructura que la versión previa basada en cajas:
        [
          {
            "class_name": <str>,
            "box": (x1, y1, x2, y2),   # bounding box del detector
            "conf": <float>,
            "crop": <np.ndarray>       # imagen recortada (warped si hay polígono; si no, crop rectangular)
          },
          ...
        ]
    En el orden dado por CLASS_NAMES, omitiendo clases no detectadas.
    """
    # 1) Inference
    model = load_model()
    results = model.predict(
        source=image_path, imgsz=IMGSZ, conf=conf_threshold, save=save, verbose=False
    )

    # 2) Leer imagen original (RGB para trabajar consistente)
    im_bgr = cv2.imread(image_path)
    if im_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    im = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

    # 3) Preparar contenedores: queremos solo la mejor detección por clase
    wanted = set(CLASS_NAMES)
    best = {cls: None for cls in wanted}  # guardamos dicts con la estructura "final"

    for r in results:
        # Si no hay cajas, nada que hacer
        if r.boxes is None or len(r.boxes) == 0:
            continue

        # Extraer arrays CPU
        cls_ids = r.boxes.cls.detach().cpu().numpy().astype(int)
        confs   = r.boxes.conf.detach().cpu().numpy().astype(float)
        boxes_xyxy = r.boxes.xyxy.detach().cpu().numpy().astype(float)

        # Máscaras (pueden ser None si el modelo devolvió solo boxes)
        segs = r.masks.xy if (hasattr(r, "masks") and r.masks is not None) else None

        for i, (cls_id, conf, xyxy) in enumerate(zip(cls_ids, confs, boxes_xyxy)):
            raw_name = model.names[int(cls_id)]
            class_name = CLASS_ALIAS.get(raw_name, raw_name)
            if class_name not in wanted:
                continue

            x1, y1, x2, y2 = map(int, xyxy)
            x1 = max(0, x1); y1 = max(0, y1)
            x2 = min(im.shape[1] - 1, x2); y2 = min(im.shape[0] - 1, y2)

            crop_img = None

            # 4) Intentar warpear usando la máscara (si existe)
            if segs is not None and i < len(segs):
                seg = np.asarray(segs[i], dtype=np.float32)  # Nx2 (x,y)
                # Sanidad: limitar a bounds de la imagen
                seg[:, 0] = np.clip(seg[:, 0], 0, im.shape[1] - 1)
                seg[:, 1] = np.clip(seg[:, 1], 0, im.shape[0] - 1)

                # Aproximar a 4 puntos y ordenar TL,TR,BR,BL
                quad = _approx_quad_or_box(seg)
                rect = _order_points(quad)

                # Dimensiones destino
                (tl, tr, br, bl) = rect
                widthA  = np.linalg.norm(br - bl)
                widthB  = np.linalg.norm(tr - tl)
                heightA = np.linalg.norm(tr - br)
                heightB = np.linalg.norm(tl - bl)
                maxW = int(max(widthA, widthB))
                maxH = int(max(heightA, heightB))

                if maxW >= 5 and maxH >= 5:
                    dst = np.array(
                        [[0, 0],
                         [maxW - 1, 0],
                         [maxW - 1, maxH - 1],
                         [0, maxH - 1]], dtype=np.float32
                    )
                    M = cv2.getPerspectiveTransform(rect, dst)
                    crop_img = cv2.warpPerspective(im, M, (maxW, maxH))

            # 5) Fallback: si no hay máscara/warping, usar crop rectangular como antes
            if crop_img is None:
                crop_img = im[y1:y2, x1:x2].copy()

            # small blur after crop
            crop_img = cv2.GaussianBlur(crop_img, (7, 7), 0)

            cand = {
                "class_name": class_name,
                "box": (x1, y1, x2, y2),
                "conf": float(conf),
                "crop": crop_img
            }

            # Guardar solo la mejor por clase
            prev = best[class_name]
            if (prev is None) or (cand["conf"] > prev["conf"]):
                best[class_name] = cand

    # 6) Devolver en el orden de CLASS_NAMES, omitiendo no detectadas
    return [best[c] for c in CLASS_NAMES if best[c] is not None]


