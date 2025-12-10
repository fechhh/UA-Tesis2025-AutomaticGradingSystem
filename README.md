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


## Guía de documentos

1- El script "1-entrenar_yolo_segment_detectar_columnas_v2.ipynb" recibe los labels desde LabelMe con las coordenadas de las columnas con celdas de respuesta en cada una de las imágenes. Este archivo las converite en formato YOLO-Seg, y utiliza estos datos para entrenar la red neuronal. La salida de este script son nuevas carpetas (una por cada imagen) con los recortes de las 3 columnas del examen.

2- El script "2-sobrepone_grillas_entrena_detector.ipynb" abre los recortes de las 3 columnas de cada examen, sobrepone una grilla sobre cada una de ellas, y recorta las celdas individuales. Estas celdas individuales son luego labeleadas manualmente en "cruz" o "no cruz", y son guardadas en carpetas separadas.

3- Dentro de la carpeta "3-modelo_detector_cruces_mobilenet" se encuentra el modelo detector de cruces, el cual recibe los recortes del script 2, y los utiliza para entrenar la red neuronal MobileNetV2.

Todos estos documentos fueron utilizados para obtener entrenar los modelos, evaluarlos mediante métricas, y hacer distintas pruebas sobre los mismos para entenderlos mejor e identificar lo que mejor funciona. Asimismo, existe una carpeta llamada "telegram_bot" donde se crearon archivos .py con las últimas versiones de los modelos ya entrenados. Estos documentos son utilizados para la corrección de exámenes mediante un bot de Telegram que guía al usuario para enviar las imágenes a evaluar y las respuestas correctas. Este sistema puede luego ser implementado en una app para celulares, sin necesidad de conexión a internet o de tener un servidor corriendo continuamente.
