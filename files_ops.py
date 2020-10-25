import glob
import os
import pathlib
import mimetypes
import cv2
import numpy as np


def get_folder_content(path: str, name=False):
    items = glob.glob(path + '/*')
    if name:
        names = [name.split('/')[-1] for name in items]
        return items, names
    return items


def get_stories(folder_name: str):
    filepath = os.path.join(pathlib.Path(os.getcwd()), folder_name)
    insta_accounts, insta_names = get_folder_content(filepath, name=True)
    insta_dict = {}
    for acc_path, acc_name in zip(insta_accounts, insta_names):
        stories = get_folder_content(acc_path)
        insta_dict[acc_name] = stories
    return insta_dict


def check_file_type(file, ftype='video'):
    mimetypes.init()
    mimestart = mimetypes.guess_type(file)[0]
    if mimestart is not None:
        mimestart = mimestart.split('/')[0]
        if mimestart == ftype:
            return True
        else:
            return False


def convert_video_to_np(video):
    cap = cv2.VideoCapture(video)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_count, frame_height, frame_width, 3), np.dtype('uint8'))
    fc = 0
    ret = True
    while fc < frame_count and ret:
        ret, buf[fc] = cap.read()
        fc += 1
    cap.release()
    return buf




