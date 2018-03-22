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
capture_ratio = 3
res = (320, 240)
fps = 30
PATH = 'data/test/'
img_name = 'kapr'
start_index = 100
img_count = 200


# initialize the cam
camera = PiCamera()
camera.resolution = res
camera.framerate = fps
rawCapture = PiRGBArray(camera, size=res)

# allow the camera to warmup
time.sleep(0.1)

# capture frames from the camera
# first sequence
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
    img_gray = img_gray.astype(float) / 255. 
    print(img_gray.dtype)   
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_gray, "NOT RECORDING",(150, 25),font, 0.5, (0,155.0,), 2, cv2.LINE_AA)
    cv2.imshow("frame", img_gray)
    key = cv2.waitKey(1) & 0xFF

    rawCapture.truncate(0)

    if key == ord('q'):
        break

print("KABEL")

# capture frames from the camera
# second sequence

i = 0
for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True):
    
    img = frame.array
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    cv2.putText(img_gray, "RECORDING",(150, 25),font, 0.5, (0,255,0), 2, cv2.LINE_AA)
    cv2.imshow("frame", img_gray)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if i % capture_ratio == 0:
        scaled_index = int(i - (i / capture_ratio) * (capture_ratio-1) + start_index)
        print(PATH + img_name + str(scaled_index) + ".jpg")
        cv2.imwrite(PATH + img_name + str(scaled_index) +  ".jpg", img_gray)
    i += 1
    if key == ord('q') or i == (img_count + start_index) * capture_ratio:
        break

