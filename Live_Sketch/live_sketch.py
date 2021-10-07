import tensorflow
import cv2
import numpy as np
import matplotlib as plt
from flask import Flask, render_template, Response

print(cv2.__version__)


def sketch(image):
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_img_blur = cv2.GaussianBlur(gray_img, (3, 3), 0)

    canny_edge = cv2.Canny(gray_img_blur, 40, 30)

    ret, mask = cv2.threshold(canny_edge, 150, 255, cv2.THRESH_BINARY_INV)
    return mask


capture = cv2.VideoCapture(0)
while True:
    success, frame = capture.read()
    cv2.imshow("Live Sketch", sketch(frame))
    if cv2.waitKey(1) == 13:
        break

capture.release()
cv2.destroyAllWindows()
