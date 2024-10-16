import os

from video_human_detection.run import run


def main():
    crowd_file = os.path.join(__file__, "crowd.mp4")
    run(
        crowd_file,
        "crowd with bboxes.mp4",
    )


if __name__ == "__main__":
    main()
