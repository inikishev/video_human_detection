import os
from collections import defaultdict
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from .render import OpenCVRenderer

__all__ = [
    "run",
]


def run(
    infile: str,
    outfile: str,
    model: str = "yolo11s.pt",
    draw_path=True,
    print_progress=True,
    device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Draws bounding boxes around people in a video using YOLO11 model. Renders a new video with the bounding boxes.

    :param infile: Path to the input video file.
    :param outfile: Path to the output video file.
    :param model: The model to predict with, can be a local file path or a model name from Ultralytics HUB.\
        The default models from ultralitics HUB are trained on the COCO dataset.\
        They can also be downloaded manually from https://docs.ultralytics.com/models/yolo11/#__tabbed_1_1
    :param print_progress: Whether to print number of video frames predicted and total number of frames. defaults to True
    :param draw_path: Whether to draw the movement path of each human on the video, defaults to True
    :param print_progress: Whether to print number of video frames predicted and total number of frames. defaults to True
    """

    # load weights
    net = YOLO(
        model=model,
        task="detect",
        verbose=False,
    ).to(device)

    # read video file
    if not os.path.isfile(infile):
        raise FileNotFoundError(f"File {infile} does not exist.")

    cap = cv2.VideoCapture(infile) # pylint:disable=E1101

    # create a renderer
    fps = cap.get(cv2.CAP_PROP_FPS) # pylint:disable=E1101
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # pylint:disable=E1101
    cur_frame = 0
    renderer = OpenCVRenderer(outfile=outfile, fps=fps)

    # track each person
    track_history = defaultdict(lambda: [])

    while cap.isOpened():
        # read frame from video
        success, frame = cap.read()

        if success:
            # detect objects in tracking mode
            results = net.track(frame, persist=True, classes=0, verbose=False)

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

            annotated_frame = results[0].plot(
                line_width=2, font_size=10, color_mode="instance"
            )  # returns numpy array

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

                    # draw paths on the image array
                    cv2.polylines(
                        annotated_frame,
                        [points],
                        isClosed=False,
                        color=(230, 230, 230),
                        thickness=5,
                    )

            # add annotated frame to renderer
            renderer.add_frame(annotated_frame)

            if print_progress:
                cur_frame += 1
                print(f"\r{cur_frame}/{frame_count}", end="\r")

        else:
            break

    # release capture and writer
    cap.release()
    renderer.release()
