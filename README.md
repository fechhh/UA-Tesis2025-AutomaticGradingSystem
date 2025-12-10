# UA-Tesis2025-AutomaticGradingSystem

Corrección robusta de exámenes de opción múltiple a partir de imágenes capturadas en condiciones reales de uso mediante técnicas de deep learning

Este repositorio contiene el código y la documentación del sistema desarrollado para la corrección automática de exámenes multiple choice a partir de imágenes.
El sistema realiza todo el proceso: desde la detección del área de burbujas hasta la predicción final de respuestas, integrando redes neuronales, técnicas de procesamiento digital de imágenes y un pipeline eficiente de inferencia.


## Características principales

✔ Entrada: Foto o escaneo de un examen (formato libre dentro de parámetros razonables, se estimó una rotación menor a 10° y perspectivas menores a 5°).

✔ Segmentación del área de burbujas: Modelo YOLO-Seg entrenado específicamente para detectar la región de respuestas.

✔ Corrección geométrica: Compensación de inclinación, perspectiva y deformaciones usando homografías.

✔ Extracción de celdas por pregunta: Recorte automático de cada burbuja según la estructura del examen.

✔ Clasificación por pregunta: Red neuronal (MobileNetV2 / ResNet-18) que determina cuál burbuja está marcada.

✔ Cálculo de confianza: Métrica interna que evalúa la consistencia de detección para identificar preguntas dudosas.

✔ Salida: Respuestas detectadas y puntaje final, opcionalmente integrable con un bot de Telegram.

