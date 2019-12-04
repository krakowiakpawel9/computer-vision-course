import cv2
import imutils

img = cv2.imread(r'images\view.jpg')
logo = cv2.imread(r'images\python.png')
logo = imutils.resize(logo, height=150)

# cv2.imshow('img', img)
# cv2.imshow('logo', logo)
# cv2.waitKey(0)

# wycięcie obszaru roi - region of interest
rows, cols, channels = logo.shape
roi = img[:rows, :cols]
# cv2.imshow('roi', roi)
# cv2.waitKey(0)

gray = cv2.cvtColor(src=logo, code=cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)

mask = cv2.threshold(src=gray, thresh=220, maxval=255, type=cv2.THRESH_BINARY)[1]
# cv2.imshow('mask', mask)

mask_inv = cv2.bitwise_not(mask)
# cv2.imshow('mask_inv', mask_inv)
# cv2.waitKey(0)

img_bg = cv2.bitwise_and(src1=roi, src2=roi, mask=mask)
logo_fg = cv2.bitwise_and(src1=logo, src2=logo, mask=mask_inv)
# cv2.imshow('img_bg', img_bg)
# cv2.imshow('logo_fg', logo_fg)


dst = cv2.add(src1=img_bg, src2=logo_fg)
img[:rows, :cols] = dst
cv2.imshow('out', img)
cv2.waitKey(0)



