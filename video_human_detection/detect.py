import os
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .render import OpenCVRenderer


def draw_people(
    in_video: str,
    out_video: str,
    weights: str = os.path.join(os.path.dirname(__file__), "yolo11n.pt"),
    draw_path=True,
):
    # create a model
    model = YOLO()

    # load weights
    weights = torch.load(weights)
    model.load(weights)

    # read video file
    video = cv2.VideoCapture(in_video)

    # create a renderer
    fps = video.get(cv2.cv.CAP_PROP_FPS)
    renderer = OpenCVRenderer(outfile=out_video, fps=fps)

    # track each person
    track_history = defaultdict(lambda: [])

    while video.isOpened():
        # read frame from video
        success, frame = video.read()

        if success:
            # detect objects in tracking mode
            results = model.track(frame, persist=True, classes=0)

            # get bounding boxes in [x, y, width, height] format.
            if results[0].boxes is None:
                raise TypeError("boxes should be a torch.Tensor, got None")
            boxes = results[0].boxes.xywh.cpu()

            # get ids of each tracked object
            if not isinstance(results[0].boxes.id, torch.Tensor):
                raise TypeError(
                    f"boxes.id should be a torch.Tensor, got {type(results[0].boxes.id)}."
                )
            track_ids = results[0].boxes.id.int().cpu().tolist()

            annotated_frame = results[0].plot() # returns numpy array

            # draw trajectories of tracked objects
            # as per official docs
            # https://docs.ultralytics.com/modes/track/#what-are-the-real-world-applications-of-multi-object-tracking-with-ultralytics-yolo
            if draw_path:
                for box, track_id in zip(boxes, track_ids):
                    x, y, _, _ = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 30:
                        track.pop(0)
                    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                    # put bounding box on the image array
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=10,
                    )

            renderer.add_frame(annotated_frame)

        else:
            break

    video.release()
