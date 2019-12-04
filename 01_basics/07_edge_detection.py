import cv2
import imutils

# Canny Edge Detection jest popularnym algorytmem do wyszukiwania krawędzi

img = cv2.imread(r'images\guido.jpg')
img = imutils.resize(image=img, height=500)
cv2.imshow('img', img)

canny = cv2.Canny(image=img, threshold1=250, threshold2=250)
cv2.imshow('canny', canny)

for thresh in [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250]:
    canny = cv2.Canny(image=img, threshold1=thresh, threshold2=thresh)
    cv2.imshow(f'canny: {thresh}', canny)
    cv2.waitKey(2000)
    cv2.destroyWindow(f'canny: {thresh}')

cv2.waitKey(0)