import os

from video_human_detection import run


def main():
    crowd_file = os.path.join(os.path.dirname(__file__), "crowd.mp4")
    run(
        crowd_file,
        "crowd with bboxes.mp4",
    )


if __name__ == "__main__":
    main()
