import cv2
import numpy as np
from helper_funcs import plot_color_image

LABELS = open("yolov3_labels.txt").read().strip().split("\n")
COLORS = np.random.randint(150, 255, size=(len(LABELS), 3), dtype="uint8")


def detect(net, image, W, H, ln, show=True):
    boxes = []
    confidences = []
    classIDs = []
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    layerOutputs = net.forward(ln)
    # Loop over each of the layer outputs
    for output in layerOutputs:
        # Loop over each of the detections
        for detection in output:
            # Extract the class ID and confidence of the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            # Filter out weak predictions by ensuring detected probability is greater some threshold
            if confidence > 0.5:
                # Scale bounding box coordinates back relative to size of image
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # Use the center (x, y)-coordinates to derive top left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # Update list of bounding box coordinates, confidences, and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    objects = []
    # Loop over the indexes and draw bounding boxes
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Extract bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # Draw a bounding box using opencv rectangle and label the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.2f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            # args: (openCVImage, resultText, (lowerLeftTextOriginX, lowerLeftTextOriginY), fontFace, fontScale, Color,
            #             fontThickness)
            objects.append((LABELS[classIDs[i]], confidences[i]))
    if show:
        # show the output image
        plot_color_image(image)
    return objects
