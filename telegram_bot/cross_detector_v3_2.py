#!/usr/bin/env python3
"""
cross_detector_v2_3.py

Integrates grid detection and cell classification to grade a single exam image.
Now also returns model confidence per cell, so you can flag low-confidence answers.

Usage in code:
```python
from cross_detector import (
    load_classification_model,
    grade_image,
    parse_key_string,
    score_answers,
)

clf_model, preprocess = load_classification_model(...)
detected = grade_image(...)
# detected: dict mapping Qn -> {'answer': int, 'error_prob': float}
key_str = "1A 2B 3A 4D"
answer_key = parse_key_string(key_str)
correct, total, details = score_answers(detected, answer_key)


############################################################
Version: 3.2
Key Changes: 
-returns all scores of each question. Formato: Qn: A=0.92, B=0.05, C=0.02, D=0.01
############################################################

```"""
import re
import cv2
import numpy as np
import torch
from typing import List, Tuple
from ultralytics import YOLO
from torchvision import transforms, models
import grid_detection_v3_3 as grid_detection

# Define grid layout: rows×cols for each grid name
GRID_SPECS = {
    "col1": (14, 4),
    "col2": (14, 4),
    "col3": (12, 4),
}

# Mapping from letter to column index
LETTER_TO_INDEX = {"A": 1, "B": 2, "C": 3, "D": 4}


def load_detection_model(weights_path: str = "runs/segment/train20/weights/best.pt") -> YOLO:
    """
    Load and return a YOLO-Seg model for grid detection.
    """
    return YOLO(weights_path)


def load_classification_model(
    model_path: str,
    img_size: int = 64,
    device: torch.device = torch.device("cpu")
) -> tuple[torch.nn.Module, transforms.Compose]:
    """
    Load the trained cell-cross classifier and return it plus the preprocessing pipeline.
    """
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 1)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    return model, preprocess


def classify_cells(
    img: np.ndarray,
    rows: int,
    cols: int,
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    device: torch.device
) -> List[Tuple[int, float, List[float]]]:
    """
    Clasifica cada celda (RGB) y devuelve, por cada fila (pregunta):
      - answer_idx: índice 1-based de la opción elegida (A=1,...)
      - row_confidence: confianza de la fila en [0,1] (= 1 - error_prob)
      - cell_scores: lista de largo 'cols' con score por cuadrado en [0,1],
                     donde score = 1 - p (p = sigmoid(logit)),
                     score alto => más probable "marcada".
    """
    h, w, _ = img.shape
    cell_h, cell_w = h / rows, w / cols

    # 1) Extract & preprocess all cells
    cells = []
    for i in range(rows):
        for j in range(cols):
            y1, y2 = int(i * cell_h), int((i + 1) * cell_h)
            x1, x2 = int(j * cell_w), int((j + 1) * cell_w)
            cell = img[y1:y2, x1:x2]
            cells.append(preprocess(cell))

    # 2) Batch inference
    batch = torch.stack(cells, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch).view(-1)
        probs = torch.sigmoid(logits).cpu().numpy()  # p ≈ "no marcada" alto, "marcada" bajo

    # 3) Por fila, calcular respuesta, confianza de fila y scores por celda
    results: List[Tuple[int, float, List[float]]] = []
    for i in range(rows):
        row_p = probs[i*cols:(i+1)*cols]              # p_j
        cell_scores = (1.0 - row_p).tolist()          # score_j = 1 - p_j  (alto => marcada)

        # Para mantener tu heurística de "error_prob" y confianza:
        order = np.argsort(row_p)
        lowest_p  = float(row_p[order[0]])
        second_p  = float(row_p[order[1]])
        total_p   = float(row_p.sum())
        low_count = int((row_p < 0.25).sum())         # umbral como en tu versión

        if low_count > 1 or low_count == 0:
            error_prob = 1.0
        else:
            gap_conf = 1.0 - (second_p - lowest_p)
            sum_conf = abs(3.0 - total_p)
            error_prob = max(gap_conf, sum_conf)
            error_prob = min(max(error_prob, 0.0), 1.0)

        answer_idx = int(order[0] + 1)                # 1-based
        row_confidence = 1.0 - error_prob

        results.append((answer_idx, row_confidence, cell_scores))

    return results


def grade_image(
    image_path: str,
    clf_model: torch.nn.Module,
    preprocess: transforms.Compose,
    device: torch.device = torch.device("cpu"),
    conf_threshold: float = 0.90
) -> dict:
    """
    Devuelve por pregunta:
      {
        "answer": int|None,
        "confidence": float,           # confianza de la fila (para compatibilidad con score_answers)
        "cell_scores": [float, float, float, float]  # score por cuadrado A..D (alto => marcada)
      }
    """
    detections = grid_detection.detect_grids(
        image_path=image_path,
        conf_threshold=conf_threshold,
        save=False
    )

    grids = {d["class_name"]: d["crop"] for d in detections if d.get("crop") is not None}

    result = {}
    q_idx = 1

    for grid_name, (rows, cols) in GRID_SPECS.items():
        img = grids.get(grid_name)

        if img is None:
            for _ in range(rows):
                result[f"Q{q_idx}"] = {
                    "answer": None,
                    "confidence": 0.0,
                    "cell_scores": [0.0] * cols
                }
                q_idx += 1
            continue

        per_row = classify_cells(img, rows, cols, clf_model, preprocess, device)
        for ans_idx, row_conf, cell_scores in per_row:
            result[f"Q{q_idx}"] = {
                "answer": ans_idx,
                "confidence": float(row_conf),
                "cell_scores": [float(s) for s in cell_scores]
            }
            q_idx += 1

    return result



def parse_key_string(key_str: str) -> dict:
    """
    Parse an answer key string like "1A 2B 3C" into a dict {"Q1":"A", ...}.
    """
    key_dict = {}
    tokens = re.split(r"\s+", key_str.strip())
    for token in tokens:
        m = re.match(r"^(\d+)([A-Da-d])$", token)
        if m:
            qnum, letter = m.groups()
            key_dict[f"Q{qnum}"] = letter.upper()
    return key_dict


def score_answers(
    detected: dict,
    answer_key: dict
) -> tuple[int, int, dict]:
    """
    Compare detected answers (with confidences) to the provided answer_key.

    Returns:
      correct_count: number of correct answers
      total: total questions in key
      details: dict mapping "Qn" -> (is_correct: bool, confidence: float)
    """
    details = {}
    correct_count = 0
    for q, letter in answer_key.items():
        entry = detected.get(q, {})
        # extract answer and confidence
        if isinstance(entry, dict):
            detected_idx = entry.get('answer')
            conf = entry.get('confidence', 0.0)
        else:
            detected_idx = entry
            conf = 0.0
        expected_idx = LETTER_TO_INDEX.get(letter)
        is_correct = (detected_idx == expected_idx)
        details[q] = {'correct': is_correct, 'confidence': conf}
        if is_correct:
            correct_count += 1
    total = len(answer_key)
    return correct_count, total, details
