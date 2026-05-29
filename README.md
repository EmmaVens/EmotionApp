<img width="391" height="868" alt="{53BC711E-47E5-42BD-BF35-464A7C11ABC4}" src="https://github.com/user-attachments/assets/ef010c8e-1261-40ea-b4bd-e77f7d3c7548" />

# Детектор эмоций

Android-приложение для анализа эмоций лица через фронтальную камеру.

## Функции

- Работа с фронтальной камерой
- Поиск лица в кадре
- Анализ эмоций через TensorFlow Lite
- Вывод основной эмоции
- Вывод top-3 результатов
- Рамка вокруг найденного лица

## Технологии

- Java
- CameraX
- ML Kit Face Detection
- TensorFlow Lite
- TFLite Interpreter

## Модель

Файл модели должен лежать здесь:

app/src/main/assets/emotion_model.tflite

## Запуск

Открыть проект в Android Studio, добавить модель и нажать Run.
