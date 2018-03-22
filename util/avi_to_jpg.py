import cv2
import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument_group()
arg.add_argument('--src', type=str, default='/home/tyson/data/avi', help="Source video directory")
arg.add_argument('--out', type=str, default='/home/tyson/data/avi-out', help="Output jpg directory")


def resize_image(image, width=376, height=376):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def capture_video(path):
    return cv2.VideoCapture(path)


def print_usage():
    parser.print_usage()


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def main(config):
    src_path = config.src
    out_path = config.out


if __name__ == "__main__":
    # Parse config
    config, unparsed = get_config()

    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)