from files_ops import get_stories, check_file_type, \
    convert_video_to_np, save_i_keyframes
from tensorflow import keras
import tensorflow as tf
from helper_funcs import plot_color_image
from tqdm import tqdm
from tensorflow.keras.models import load_model
from load_img import load_image_pixels, load_pixels
from object_det_yolo import decode_netout, correct_yolo_boxes, do_nms, draw_boxes, get_boxes


LABELS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

ANCHORS = [[116, 90, 156, 198, 373, 326], [30, 61, 62, 45, 59, 119], [10, 13, 16, 30, 33, 23]]
class_threshold = 0.5


if __name__ == '__main__':
    # load yolov3 model
    model = load_model('model.h5')
    # define the expected input shape for the model
    input_w, input_h = 416, 416
    # define our new photo

    # make prediction
    folder_name = 'insta_stories'
    insta_dict = get_stories(folder_name)
    for key, vals in insta_dict.items():
        print(key)
        for val in vals:
            if check_file_type(val):
                images, text = save_i_keyframes(val)  # images = convert_video_to_np(val)
                for im in images:
                    image, image_w, image_h = load_pixels(im, (input_w, input_h))
                    yhat = model.predict(image)
                    boxes = list()
                    for i in range(len(yhat)):
                        # decode the output of the network
                        boxes += decode_netout(yhat[i][0], ANCHORS[i], class_threshold, input_h, input_w)
                    # correct the sizes of the bounding boxes for the shape of the image
                    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)
                    # suppress non-maximal boxes
                    do_nms(boxes, 0.3)
                    # get the details of the detected objects
                    v_boxes, v_labels, v_scores = get_boxes(boxes, LABELS, class_threshold)
                    # summarize what we found
                    for i in range(len(v_boxes)):
                        print(v_labels[i], v_scores[i])
                    # draw what we found
                    draw_boxes(im, v_boxes, v_labels, v_scores, image=False)







