import numpy as np
from numpy.linalg import norm
import cv2
from skimage.filters import threshold_local
import argparse
import imutils
import sys


def get_perspective(image, contours, ratio):
    """
    This function takes image and contours and returns perspective of this contours.
    :param image: image, numpy array
    :param contours: contours, numpy array
    :param ratio: rescaling parameter to the original image
    :return: warped image
    """
    points = contours.reshape(4, 2)
    points = points * ratio
    rectangle = np.zeros(shape=(4, 2), dtype='float32')

    total = points.sum(axis=1)
    rectangle[0] = points[np.argmin(total)]
    rectangle[2] = points[np.argmax(total)]

    difference = np.diff(points, axis=1)
    rectangle[1] = points[np.argmin(difference)]
    rectangle[3] = points[np.argmax(difference)]

    # rectangle *= ratio

    (a, b, c, d) = rectangle
    width1 = norm(c - d)
    width2 = norm(b - a)

    height1 = norm(b - c)
    height2 = norm(a - d)

    max_width = max(int(width1), int(width2))
    max_height = max(int(height1), int(height2))

    destination = np.array([[0, 0],
                            [max_width - 1, 0],
                            [max_width - 1, max_height - 1],
                            [0, max_height - 1]], dtype='float32')

    M = cv2.getPerspectiveTransform(src=rectangle, dst=destination)
    warped_image = cv2.warpPerspective(src=image, M=M, dsize=(max_width, max_height))
    return warped_image


ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to image')
args = vars(ap.parse_args())

print(f'Numpy version: {np.__version__}')
print(f'OpenCV version: {cv2.__version__}')

# wczytanie obrazu z dokumentem
image = cv2.imread(args['image'])
original_image = image.copy()
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height=500)

# konwersja do skali szarości
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray_image', gray_image)

# rozmycie obrazu
gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
cv2.imshow('gray_image_blurred', gray_image)

# detekcja krawędzi
edges = cv2.Canny(gray_image, 20, 20)

cv2.imshow('image', image)
cv2.imshow('edges', edges)

# znalezienie konturu dokumentu
contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

screen_contour = None

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:
        screen_contour = approx
        break

if not isinstance(screen_contour, np.ndarray):
    print('Nie można przetworzyć obrazu.')
    sys.exit(1)

image_v = cv2.drawContours(image.copy(), screen_contour, -1, (0, 255, 0), 10)
cv2.imshow('outline_v', image_v)

image_cnt = cv2.drawContours(image.copy(), [screen_contour], -1, (0, 255, 0), 3)
cv2.imshow('outline_cnt', image_cnt)

# ekstrakcja perspektywy
warped_image = get_perspective(original_image, screen_contour, ratio)
cv2.imshow('out', warped_image)

# konwersja do skali szarości
warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)

# obliczenie maski progowej na podstawie sąsiedztwa pikseli
T = threshold_local(image=warped_image, block_size=11, offset=10, method='gaussian')
warped_image = (warped_image > T).astype('uint8') * 255
cv2.imshow('warped_image', warped_image)
cv2.waitKey(0)