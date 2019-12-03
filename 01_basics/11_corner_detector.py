import cv2
import numpy as np
import imutils

img = cv2.imread(r'assets\cube.jpg')
img = imutils.resize(img, height=600)
# cv2.imshow('img', img)

gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)

gray = np.float32(gray)
dst = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.dilate(src=dst, kernel=None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]

cv2.imshow('dst', img)
cv2.waitKey(0)