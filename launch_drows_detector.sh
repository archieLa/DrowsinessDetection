#!/bin/bash
# Go to drows detection folder
cd ~/Documents/Sleep_Detection/DrowsinessDetection
# Launch the program
python SleepDetector.py --cascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat --warning-sound warning.wav > log.txt &



