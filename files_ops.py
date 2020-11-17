import glob
import os
import pathlib
import mimetypes
import cv2
import numpy as np
import subprocess
import pytesseract
from helper_funcs import remove_special_chars
import pdb


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


def get_frame_types(video_fn):
    command = 'ffprobe -v error -show_entries frame=pict_type -of default=noprint_wrappers=1'.split()
    out = subprocess.check_output(command + [video_fn]).decode()
    frame_types = out.replace('pict_type=','').split()
    return zip(range(len(frame_types)), frame_types)


def save_i_keyframes(video_fn):
    frame_types = get_frame_types(video_fn)
    i_frames = [x[0] for x in frame_types if x[1] == 'I']
    full_frame = []
    frame_text = []
    if i_frames:
        cap = cv2.VideoCapture(video_fn)
        for frame_no in i_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            text_str = pytesseract.image_to_string(frame_rgb, lang="eng+rus")
            text_str = remove_special_chars(text_str)
            if len(text_str) > 0:
                frame_text.extend(text_str)
            full_frame.append(frame_rgb)
        cap.release()
        return np.asarray(full_frame), frame_text
    else:
        print('No I-frames in '+ video_fn)




