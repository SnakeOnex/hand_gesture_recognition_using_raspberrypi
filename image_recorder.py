# importing stuff
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import numpy as np
import tensorflow as tf
import cv2
import math

def resize_img(img, img_size):
    y = math.floor((img_size/img.shape[0]) * img.shape[1])
    resized = cv2.resize(img, (y, img_size))
    return resized
   
def crop_img(img, final_width):
    left_border = math.floor((img.shape[1] - final_width) / 2)
    right_border = final_width + left_border
    img_cropped = img[:, left_border:right_border]
    return img_cropped


# params
img_size = 64
recognition_ratio = 3
res = (320, 240)
fps = 30


# initialize the cam
camera = PiCamera()
camera.resolution = res
camera.framerate = fps
rawCapture = PiRGBArray(camera, size=res)

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):

    # current frame as numpy array 
    img = frame.array
    # print(img.shape)

    # converts frame to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img_gray.shape)
   
    img_resized = resize_img(img_gray, img_size) 
     #print(img_resized.shape) 
   
    img_cropped = crop_img(img_resized, img_size)
    print(img_cropped.shape)
    cv2.imshow("frame", img_gray)
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord('q'):
        break

print("KABEL")

# capture frames from the camera
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    
    img = frame.array
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    cv2.imshow("frame", img_gray)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)

    if key == ord('q'):
        break

