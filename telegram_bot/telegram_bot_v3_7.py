#!/usr/bin/env python3
"""
telegram_bot_v3_7.py

A Telegram bot that grades multiple-choice exam images by integrating the cross_detector pipeline.
Includes debug prints of detected answers and answer keys, and flags low-confidence questions.

Workflow:
1. User sends /start to see instructions.
2. User sends an answer key in the format: "1A 2B 3C ..." (no slash command needed).
3. User sends a photo of their exam grid.
4. Bot responds with detected answers, the key, number of correct answers, which questions were wrong,
   and warns about any low-confidence detections.
5. User can send /quit to clear the key and start over.

Configuration:
- Pass the bot token via `--token` or set TELEGRAM_TOKEN env var.

############################################################
Version: 3.7
Key Changes:
- Inform user when more than one answer is detected for a question.
- In that case, consider it as incorrect.
- If >4 answers with low confidence, warn user to retake photo.
- Considers if the answer key has less than 40 questions and only grades those.
############################################################
"""
import os
import tempfile
import argparse
import logging
import torch
import numpy as np
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from cross_detector_v3_2 import (
    load_classification_model,
    grade_image,
    parse_key_string,
    score_answers,
)

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Globals to hold models and preprocessing
clf_model = None
preprocess = None
device = torch.device("cpu")
# Store per-chat answer keys
answer_keys: dict[int, dict] = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "👋 Welcome to ExamGraderBot!\n"
        "Send the answer key (e.g.: `1A 2B 3C 4D`), then send your exam photo to grade.\n"
        "Type /quit to reset the key at any time."
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "/start - Show instructions\n"
        "/help - Show this help message\n"
        "/quit - Clear the current answer key\n"
        "Send answer key as text (e.g. `1A 2B 3C`), then send the photo."
    )

async def tips_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Provide photography tips to the user.
    """
    tips = [
        "• Place the exam grid flat and straight in the frame.",
        "• Ensure even, bright lighting—avoid shadows and glare.",
        "• Hold the camera parallel to the paper, not at an angle.",
        "• Use high resolution; avoid digital zoom or heavy compression.",
        "• Keep the camera steady; use both hands or a tripod if needed.",
    ]
    await update.message.reply_text("📸 Photo Tips:\n" + "\n".join(tips))

async def quit_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id in answer_keys:
        answer_keys.pop(chat_id)
        await update.message.reply_text(
            "🗑️ Answer key cleared. Send a new key to start grading."
        )
    else:
        await update.message.reply_text(
            "⚠️ No answer key set. Send a key first, like: 1A 2B 3C ..."
        )

async def set_key(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()
    if text.startswith("/"):
        return
    key = parse_key_string(text)
    chat_id = update.effective_chat.id
    if not key:
        await update.message.reply_text(
            "❌ Could not parse key. Use format: 1A 2B 3C ..."
        )
    else:
        answer_keys[chat_id] = key
        await update.message.reply_text(
            f"✅ Answer key set for {len(key)} questions.\n"
            "Now send your exam image to grade, or use /quit to reset."
        )

def compute_question_confidence(cell_scores: list[float]) -> float:
    s = np.clip(np.array(cell_scores, dtype=float), 0.0, 1.0)
    # 100% si es exactamente one-hot
    if np.isclose(s.max(), 1.0) and np.isclose(s.sum(), 1.0):
        return 1.0
    if float(s.max()) == 0.0:
        return 0.0  # blanco total

    s_sorted = np.sort(s)
    s1 = float(s_sorted[-1])
    s2 = float(s_sorted[-2]) if len(s_sorted) > 1 else 0.0

    margin_conf = s1 - s2                                  # 0..1
    sum_conf    = 1.0 - min(abs(float(s.sum()) - 1.0), 1.0) # 1 si sum≈1
    peak_conf   = s1

    conf = (margin_conf + sum_conf + peak_conf) / 3.0
    return float(np.clip(conf, 0.0, 1.0))


LOW_CONFIDENCE_THRESHOLD = 0.75  # umbral para advertir al usuario de baja confianza en una pregunta

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if chat_id not in answer_keys:
        await update.message.reply_text(
            "⚠️ Please send the answer key first, like: 1A 2B 3C ..."
        )
        return

    await update.message.reply_text("🕑 Grading your exam, please wait…")

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        image_path = tf.name
    await file.download_to_drive(image_path)

    try:
        detected = grade_image(
            image_path,
            clf_model,
            preprocess,
            device=device,
            conf_threshold=0.90,
        )
    except Exception:
        logger.exception("Grading failed")
        await update.message.reply_text(
            "❌ Error grading image. Please send a clearer photo or re-send the key."
        )
        os.remove(image_path)
        return

    # --- limitar la corrección a la cantidad de preguntas provistas en la key ---
    answer_key = answer_keys[chat_id]
    n_key = len(answer_key)
    if n_key > 0:
        # Mantener únicamente las primeras n_key preguntas por orden numérico (Q1, Q2, ...)
        ordered_qs = sorted(detected.keys(), key=lambda k: int(k[1:]))
        keep = set(ordered_qs[:n_key])
        detected = {q: detected[q] for q in ordered_qs if q in keep}


    choice_map = ['A', 'B', 'C', 'D']
    LOW_CONFIDENCE_THRESHOLD = 0.50
    MULTI_MARK_THRESHOLD = 0.8        # score para considerar “marcada”

    # --- Calcular confianza por pregunta y recopilar solo las de baja confianza ---
    low_conf_questions = []
    for q in sorted(detected.keys(), key=lambda k: int(k[1:])):
        cs = detected[q].get("cell_scores", []) or []
        if cs and max(cs) > 0:
            ans_idx = int(np.argmax(cs) + 1)
            conf_q = compute_question_confidence(cs)
        else:
            ans_idx = None
            conf_q = 0.0

        detected[q]["confidence"] = float(conf_q)  # para score_answers más adelante

        # Detectar múltiples marcas (>= 2 celdas con score alto)
        MULTI_MARK_THRESHOLD = 0.8 

        multi_count = sum(1 for s in cs if s >= MULTI_MARK_THRESHOLD)
        multi_note = ""
        if multi_count > 1:
            multi_note = " [⚠️ multiple answers detected]"
            # >>> Invalida la pregunta para scoring:
            detected[q]["answer"] = None
            # (opcional) marca explícita por si querés usarlo luego
            detected[q]["multi_mark"] = True
        else:
            detected[q]["answer"] = ans_idx
            detected[q]["multi_mark"] = False

        # Sólo agregamos a la lista de baja confianza si cae bajo el umbral
        if conf_q < LOW_CONFIDENCE_THRESHOLD:
            if detected[q]["answer"] is None:
                low_conf_questions.append(
                    f"{q}: no detected answer ({conf_q:.0%} confidence){multi_note}\n"
                )
            else:
                ans_letter = choice_map[detected[q]['answer'] - 1]
                low_conf_questions.append(
                    f"{q}: detected answer {ans_letter} ({conf_q:.0%} confidence){multi_note}\n"
                )

    # --- Corte temprano: si hay 5 o mas de baja confianza, no mostramos nada más ---
    if len(low_conf_questions) >= 5:   # “mas de 4” => 5 o mas
        await update.message.reply_text(
            "⚠️ I'm detecting a poor quality photo.\n"
            "Please retake the photo following /tips."
        )
        os.remove(image_path)
        return

    # --- Si paso el corte, ahora si calculamos el score y mostramos resultados ---
    answer_key = answer_keys[chat_id]
    correct, total, details = score_answers(detected, answer_key)

    msg_lines = [f"📝 Score: {correct}/{total} correct."]

    if low_conf_questions:
        msg_lines.append("⚠️ Questions with low confidence. Please check:")
        msg_lines.extend(low_conf_questions)

    msg_lines.append("\n➡️ Send the next exam image to grade, or type /quit to finish.")
    await update.message.reply_text("\n".join(msg_lines))

    os.remove(image_path)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--token", default=os.getenv("TELEGRAM_TOKEN"),
        help="Telegram bot token or set TELEGRAM_TOKEN",
    )
    parser.add_argument(
        "--det-weights", default="runs/detect/train23/weights/best.pt",
        help="YOLO detection weights",
    )
    parser.add_argument(
        "--clf-model", default="cell_cross_classifier.pt",
        help="Cell classifier model",
    )
    parser.add_argument(
        "--img-size", type=int, default=64,
        help="Classifier image size",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use GPU if available",
    )
    args = parser.parse_args()

    if not args.token:
        raise ValueError("Token required: use --token or TELEGRAM_TOKEN env var")

    global clf_model, preprocess, device
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")

    # Load models
    clf_model, preprocess = load_classification_model(
        args.clf_model, img_size=args.img_size, device=device
    )

    # Start bot
    app = ApplicationBuilder().token(args.token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("tips", tips_command))
    app.add_handler(CommandHandler("quit", quit_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, set_key))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    app.run_polling()

if __name__ == "__main__":
    main()
