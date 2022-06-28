import cv2
import numpy as np
kernel_size= (5,5)

def detect_edges(frame):
    # filter for yellow lane lines
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv,(5,5),0)
    lower_yellow = np.array([0,  0,  124])
    upper_yellow = np.array([55, 255, 255])
    mask = cv2.inRange(blur, lower_yellow, upper_yellow)
    mask = cv2.erode(mask,kernel_size)
    mask = cv2.dilate(mask,kernel_size)
    edges = cv2.Canny(mask, 300, 500)
    return edges

def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)

    # only focus bottom half of the screen
    polygon = np.array([[
        (0, height * 1 / 2),
        (width, height * 1 / 2),
        (width, height),
        (0, height),
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges