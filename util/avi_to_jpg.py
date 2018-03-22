import os
import argparse
import cv2

parser = argparse.ArgumentParser()
arg = parser.add_argument_group()
arg.add_argument('--src', type=str, default='/home/tyson/data/avi', help="Source video directory")
arg.add_argument('--out', type=str, default='/home/tyson/data/avi-out', help="Output jpg directory")
arg.add_argument('--transform', type=str, choices=('resize', 'crop'), default='crop',
                 help='Resize or crop the image to size')


def get_video_list(path):
    assert os.path.exists(path), f'{path} is not a valid path'

    is_avi = lambda f: os.path.isfile(f'{path}/{f}') and len(f) > 4 and f[-4:] == '.avi'
    files = [file for file in os.listdir(path) if is_avi(file)]
    return files


def capture_video(path):
    return cv2.VideoCapture(path)


def save_frames(video, video_fname, config):
    out = config.out
    if config.transform == 'crop':
        transform = crop_image
    else:
        transform = resize_image

    frame = 0
    success, img = video.read()
    while success:
        img = transform(img)
        out_name = f'{out}/{video_fname[:-4]}_frame{frame}.jpg'
        cv2.imwrite(out_name, img)
        frame += 1

        success, img = video.read()


def resize_image(image, width=376, height=376):
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def crop_image(image, width=376, height=376):
    base_h, base_w = image.shape[:2]
    y1 = int((base_h - height)/2)
    y2 = int(height + y1)
    x1 = int((base_w - width)/2)
    x2 = int(width + x1)
    return image[y1:y2, x1:x2]


def print_usage():
    parser.print_usage()


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def main(config):
    src_path = config.src

    for video_path in get_video_list(src_path):
        video = capture_video(f'{src_path}/{video_path}')
        print('Saving frames for ', video_path, ' in ', config.out)
        save_frames(video, video_path, config)
        print('Saved frames for ', video_path)


if __name__ == "__main__":
    # Parse config
    config, unparsed = get_config()

    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)