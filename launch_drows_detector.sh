#!/bin/bash
# Go to drows detection folder
cd ~/Documents/Sleep_Detection/DrowsinessDetection
# Launch the program
python Source/SleepDetector.py --cascade RequiredFiles/haarcascade_frontalface_default.xml --shape-predictor RequiredFiles/shape_predictor_68_face_landmarks.dat --warning-sound RequiredFiles/warning.wav > log.txt &



