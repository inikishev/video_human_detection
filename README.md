# Описаниe проекта
Отрисовывает рамки (bounding boxes) вокруг людей на видеозаписи при помощи [YOLO11](https://github.com/ultralytics/ultralytics).

Пример:
![{089BF6F8-62BD-4CDA-B412-82B492C2C315}](https://github.com/user-attachments/assets/1798fd14-5184-46ae-8811-83c169389a62)

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

По умолчанию используется [YOLO11s, обученная на COCO](https://docs.ultralytics.com/models/yolo11/#performance-metrics), которая будет загружена при первом запуске (загрузится 19 мб).
Можно использовать другую модель с [Ultralitics HUB](https://docs.ultralytics.com/models/) либо загруженную локально:
```py
from video_human_detection import run
run(
  infile = 'path/to/input.mp4',
  outfile = 'path/to/output.mp4',
  model = 'yolo11n.pt', # название предобучнной модели с Ultralitics HUB либо путь к загруженной локально модели
)
```
