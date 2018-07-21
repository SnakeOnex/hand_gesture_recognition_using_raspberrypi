# Hand Gesture Recognition with Raspberry Pi

My first Machine Learning project I made in the 3rd year of high school.

## Introduction

The project goes over the whole process of training a deep learning model for recognising hand gestures. (or anything else actually)

## 1. Getting data

The **image_recorder.py** file serves the purpose of recording a lot of images in a short time.

## 2. Training the model

Files **like_dislike.py**, **rock_paper_scissors.py** and **fingers.py** serve the purpose of training a tensorflow model to recognise their individual set of images.

## 3. Recognising gestures from live video

Files **\<gesture-name\>**_recognition.py are for live recognition (using the trained models from step 2) of certain gesture from live video feed from RaspberryPi Camera.

## Used Technologies

- Tensorflow
- OpenCV
- scipy


## Video demonstration

[![Video demonstration](https://i.imgur.com/O9x4vO9.png)](https://www.youtube.com/watch?v=OBWjdnMJSWE)

