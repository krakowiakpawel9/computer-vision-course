import pytesseract
from PIL import Image
import imutils
import cv2


# https://github.com/UB-Mannheim/tesseract/wiki
# https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-v5.0.0-alpha.20191030.exe
# pip install pytesseract
# Add Tesseract to the PATH
# C:\Program Files\Tesseract-OCR

def ocr(filename):
    return pytesseract.image_to_string(image=Image.open(filename))


filename = r'cap_4.jpg'
img = cv2.imread(filename)

print(ocr(filename))

cv2.imshow('img', img)
cv2.waitKey(0)