from files_ops import get_stories, check_file_type, save_i_keyframes
import cv2
from detect_objs import detect

net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yolov3.weights')

if __name__ == '__main__':
    # define the expected input shape for the model
    input_w, input_h = 416, 416

    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()

    folder_name = 'insta_stories'
    insta_dict = get_stories(folder_name)
    for key, vals in insta_dict.items():
        print(key)
        for val in vals:
            if check_file_type(val):
                images, text = save_i_keyframes(val)
                for im in images:
                    H, W = im.shape[:-1]
                    blob = cv2.dnn.blobFromImage(im, 1 / 255.0, (input_h, input_w),
                                                 swapRB=True, crop=False)
                    net.setInput(blob)
                    print(detect(net, im, W, H, ln, show=True))







