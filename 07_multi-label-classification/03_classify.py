from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import pickle
import imutils
import argparse
import cv2
import os

# suppress logs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# przykładowe uruchomienie
# py 03_classify.py -i downloads\black_dress\0000.jpg -m output\model_04_12_2019_16_12.hdf5

def load(filename):
    image = cv2.imread(filename)
    image = cv2.resize(image, (150, 150))
    image = image.astype('float') / 255.
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image')
ap.add_argument('-m', '--model', required=True, help='path to model')
args = vars(ap.parse_args())

print('[INFO] Loading model...')
model = load_model(args['model'])
image = load(args['image'])
y_pred = model.predict(image)[0]

print('[INFO] Loading labels...')
with open(r'output\mlb.pickle', 'rb') as file:
    mlb = pickle.loads(file.read())

labels = dict(enumerate(mlb.classes_))
idxs = np.argsort(y_pred)[::-1]

print('[INFO] Loading image...')
image = cv2.imread(args['image'])
image = imutils.resize(image, width=1000)

print('[INFO] Displaying image...')
for i, idx in enumerate(idxs[:2]):
    cv2.putText(img=image, text=f'Labels: {labels[idx]:6} Probability: {y_pred[idx] * 100:.4f}%',
                org=(10, (i * 30) + 25), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.8,
                color=(0, 179, 137), thickness=2)

cv2.imshow('image', image)
cv2.waitKey(0)
