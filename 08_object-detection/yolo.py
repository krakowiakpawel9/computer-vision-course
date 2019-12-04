import cv2
import numpy as np
import os
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image')
args = vars(ap.parse_args())

np.random.seed(10)

CONFIDENCE = 0.5
THRESHOLD = 0.3

print('[INFO] Loading labels...')
labels_path = os.path.join('yolo', 'coco.names')
LABELS = open(labels_path).read().strip().split('\n')

COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

weights_path = os.path.join('yolo', 'yolov3.weights')
config_path = os.path.join('yolo', 'yolov3.cfg')

net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

image = cv2.imread(args['image'])
cv2.imshow('img', image)
(h, w) = image.shape[:2]

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

blob = cv2.dnn.blobFromImage(image, 1/255., (416, 416), swapRB=True, crop=False)
net.setInput(blob)
start = time.time()
layer_outputs = net.forward(ln)
end = time.time()

boxes = []
confidences = []
class_ids = []

for output in layer_outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONFIDENCE:
            box = detection[0:4] * np.array([w, h, w, h])
            (x_center, y_center, width, height) = box.astype('int')

            x = int(x_center - (width / 2))
            y = int(y_center - (height / 2))

            boxes.append([x, y, int(width), int(height)])
            confidences.append(float(confidence))
            class_ids.append(class_id)

idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

if len(idxs) > 0:
    for i in idxs.flatten():

        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])

        color = [int(c) for c in COLORS[class_ids[i]]]

        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = f'{LABELS[class_ids[i]]}: {confidences[i]:.4f}'
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow('image', image)
cv2.waitKey(0)