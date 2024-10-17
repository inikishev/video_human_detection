# Описаниe проекта
Отрисовывает рамки (bounding boxes) вокруг людей на видеозаписи при помощи [YOLO11](https://github.com/ultralytics/ultralytics).

Пример:
![image](https://github.com/user-attachments/assets/49699a40-ee0f-477a-93d9-475235223cfd)

# Установка
Установка для pip и conda:
```
pip install -e git+https://github.com/inikishev/video_human_detection.git#egg=video_human_detection
```
Код тестировался на Python 3.12.

# Запуск
Функция `run` считывает выбранный видеофайл `infile` и записывает новый видеофайл `outfile` с отрисованными рамками.
```py
from video_human_detection import run
run(infile = 'path/to/input.mp4', outfile = 'path/to/output.mp4')
```

По умолчанию используется [YOLO11, обученная на COCO](https://docs.ultralytics.com/models/yolo11/#performance-metrics). Можно использовать другую модель с [Ultralitics HUB](https://docs.ultralytics.com/models/), и другие веса:
```py
from video_human_detection import run
run(
  infile = 'path/to/input.mp4',
  outfile = 'path/to/output.mp4',
  model = 'yolo11n.pt', # название модели с Ultralitics HUB либо путь к загруженной локально модели
  weights = 'path/to/weights.pt', # путь к кастомным весам
)
```
