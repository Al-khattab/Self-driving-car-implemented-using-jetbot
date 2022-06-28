import cv2
import numpy as np
import logging
import math
import datetime
import sys
from mask import *
from lines import *
from steering_angles import *
from jetcam.csi_camera import CSICamera
from jetbot import Robot
import torch
import torchvision
import torch.nn.functional as F

print("Initializing camera")
cap = camera = CSICamera(width=224, height=224, capture_fps=10)

print("Initializing parameters")
curr_angle = 90
new_angle = 90
gain = 0.03
left_value = 0
right_value = 0
robot = Robot()

print("Initializing detection model")
model = torchvision.models.alexnet(pretrained=False)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 2)
model.load_state_dict(torch.load('best_model.pth'))
device = torch.device('cuda')
model = model.to(device)
print("Model loaded")

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])
normalize = torchvision.transforms.Normalize(mean, stdev)

def detect_lane(frame):
    edges = detect_edges(frame)
    cropped_edges = region_of_interest(edges)
    line_segments = detect_line_segments(cropped_edges)
    lane_lines = average_slope_intercept(frame, line_segments)
    return lane_lines

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

#display functions
def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def display_heading_line(frame, steering_angle, line_color=(0, 0, 255), line_width=5 ):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    steering_angle_radian = steering_angle / 180.0 * math.pi
    x1 = int(width / 2)
    y1 = height
    x2 = int(x1 - height / 2 / math.tan(steering_angle_radian))
    y2 = int(height / 2)
    cv2.line(heading_image, (x1, y1), (x2, y2), line_color, line_width)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    return heading_image


while True:
    frame = cap.read()
    x = preprocess(frame)
    y = model(x)
    y = F.softmax(y, dim=1)
    prob_blocked = float(y.flatten()[0])
    line_image= detect_lane(frame)
    stream = display_lines(frame,line_image)
    new_angle = compute_steering_angle(frame,line_image)
    curr_angle = stabilize_steering_angle(curr_angle, new_angle, len(line_image))
    deviation = curr_angle - 90
# movment control
    print("adjusting movment")
    if prob_blocked > 0.6:
        left_value = 0
        right_value = 0
        
    elif prob_blocked < 0.6:
        if deviation < 5 and deviation > -5:
            left_value = 0.11
            right_value = 0.11
        
        elif deviation > 4:
            left_value = 0.11 + gain
            right_value = 0.11 - gain
        
        elif deviation < -4:
            left_value = 0.11 - gain
            right_value = 0.11 + gain
   
        if len(line_image) == 0:
            left_value = 0
            right_value = 0
        
    robot.left_motor.value = left_value
    robot.right_motor.value = right_value
    #display options
    mask = detect_edges(frame)
    heading_line = display_heading_line(stream,curr_angle)
    #cv2.imshow('frame',heading_line)
    #cv2.imshow('mask',mask)
    #print(deviation)
    print(prob_blocked)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

print("Demo end")
cap.realase()
cv2.destroyAllWindows()
