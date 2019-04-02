from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
import os
import datetime as dt
import argparse
import time
import dlib
import cv2
import pygame
import tkinter 
import PIL 
from PIL import ImageTk




class WarningAnnouncer:
    def __init__(self, args):
        pygame.mixer.init()
        pygame.mixer.music.load("warning.wav")
        self.alreadyTrigerred = False

    def warn(self):
        # Only need to start the warning sound once
        # And then it will be on untill it is stopped
        # Don't want to retrigger multiple times 
        if (self.alreadyTriggered == False):
            print("Warning triggered")
            pygame.mixer.music.play(-1)
            self.alreadyTriggered = True            

    def stop_warning(self):
        pygame.mixer.music.stop()
        self.alreadyTriggered = False    


class DrowsAnalyst:
    def __init__(self, args, updateInterMiliSec):
        # Timer interval of frame updates in sec
        self.timeInter = updateInterMiliSec/1000
        print("Setting up face and landmark detectors")
        # Set up face detector
        self.detector = cv2.CascadeClassifier(args["cascade"])
        # Set up landmark detector
        self.predictor = dlib.shape_predictor(args["shape_predictor"])
        # Set up constants
        # EAR threshold to determine if eyes are closed
        self.EAR_THRESH = 0.22
        # Threshold to determine if user has fallen asleep sec
        self.ASLEEP_THRESH = 5
        # Use imutils to get array indexes of start and end of each eye
        (self.lEyeStart, self.lEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rEyeStart, self.rEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        # Helper variables
        self.eyesClosed = False
        self.isUserAsleep = False
        self.timeStartEyesOpen = dt.datetime.now()
        self.timeStartEyesClosed = dt.datetime.now()
        self.currentBlinkCounter = 0.0
        self.prevBlinkCounter = 0.0
        self.minuteCompleted = False
        self.currentMaxEyesClosed = 0.0
        self.currentMinEyesClosed = 0.0
        self.prevMaxEyesClosed = 0.0
        self.prevMinEyesClosed = 0.0
        self.timeEyesClosed = 0.0
        self.currentSumEyesClosed = 0.0
        self.prevSumEyesClosed = 0.0
        self.firstFrame = True
        self.minuteCounter = 0.0
    
    def ear_calc(self, eye):
        # Compute euclidean distance between sets of vertical
        # points indicating eyes from the facial landmark
        # Based on the facial landmark we know which points indicate eyes
        # exactly

        ver_dist1 = dist.euclidean(eye[1], eye[5])
        ver_dist2 = dist.euclidean(eye[2], eye[4])

        # Compute euclidean distance for horizontal point of the eye
        hor_dist = dist.euclidean(eye[0], eye[3])

        # Now calculate EAR
        ear = (ver_dist1 + ver_dist2) / (2.0 * hor_dist)

        return ear

    def provide_drows_data(self, frame, drowsData):
        
        # If this is the first sample 
        if (self.firstFrame == True):   # Frame received for the first time so we want to save the time, this should only execute once
            self.oneMinuteTimerStart = dt.datetime.now()
            self.firstFrame = False
        
        # Resize image for efficiency
        frame = imutils.resize(frame, width = 300)
        # Convert received frame to gray scale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Use detector to detect faces in the image
        # Please note that there should be only one face detected
        # when the device is used in the vehicle (only one driver)
        # In case we can't detect the face we clear all the data
        faces = self.detector.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5,
                                          minSize = (30,30), flags = cv2.CASCADE_SCALE_IMAGE)
        # If face was not detected zero out the data and return unmodified video frame
        if (len(faces) == 0):
            drowsData['EAR'] = 0.0
            drowsData['ETC'] = 0.0
            drowsData['ETO'] = 0.0
            drowsData['BPM'] = 0.0
            drowsData['DBPM'] = 0.0
            drowsData['ATCPM'] = 0.0
            drowsData['DATCPM'] = 0.0
            drowsData['MAXTCPM'] = 0.0
            drowsData['MINTCPM'] = 0.0
            drowsData['DLEVEL'] = 'LOW'
            return frame

        # Face was detected so we can proceed
        ear = 0
        (x, y, w, h) = faces[0]
        # Create a rectangle around the detected face and provide
        # its coordinates to dlib facial landmark predictor
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        face_landmark = self.predictor(gray, rect)
        face_landmark = face_utils.shape_to_np(face_landmark)
        
        # Retrieve coord of left and eye right in the face 
        leftEye = face_landmark[self.lEyeStart:self.lEyeEnd]
        rightEye = face_landmark[self.rEyeStart:self.rEyeEnd]
        # Calculate ear for each eye
        leftEar = self.ear_calc(leftEye)
        rightEar = self.ear_calc(rightEye)
        # Calculate combined EAR
        ear =  (leftEar + rightEar) / 2.0

        # Use convex hull for marking eyes in the video stream
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 0, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 0, 0), 1)
        
        # Find if eyes are closed
        if (ear < self.EAR_THRESH):    # User has closed eyes
            if (self.eyesClosed == False):  # Previously user had eyes opened
                self.timeStartEyesClosed = dt.datetime.now() #Mark the time of when user closed the eyes  
                self.eyesClosed = True
                # We also want to include the time between last eyes time opened and now closed
                # The assumption we are making is that if the eyes were previously opened and now are closed
                # then during the time between open and closed they were opened so it is not completely real time
                # but it is good enough approx
                self.timeDiffInSec = ((self.timeStartEyesClosed - self.timeStartEyesOpen).microseconds) / 1e6
                drowsData['ETO'] += self.timeDiffInSec
            elif (self.eyesClosed == True):
                timeNow = dt.datetime.now()
                self.timeDiffInSec = ((timeNow - self.timeStartEyesClosed).microseconds) / 1e6
                self.timeStartEyesClosed = timeNow
                self.timeEyesClosed += self.timeDiffInSec
                self.check_if_user_asleep(self.timeEyesClosed)
                drowsData['ETC'] = self.timeEyesClosed # Provide the time eyes have been closed
                drowsData['ETO'] = 0.0
        else: # User has opened eyes
            if (self.eyesClosed == True):   # User had closed eyes and now opened - blinked
                self.eyesClosed = False
                self.isUserAsleep = False
                self.currentBlinkCounter += 1 # count the blink
                self.timeStartEyesOpen = dt.datetime.now() # Mark the time of when user opened the eyes
                # We also want to include the time between last eyes time closed and now opened
                # The assumption we are making is that if the eyes were previously closed and now are open
                # then during the time between closed and open they were opened so it is not completely real time
                # Bit in this case it is good enough approx
                self.timeDiffInSec = ((self.timeStartEyesOpen - self.timeStartEyesClosed).microseconds) / 1e6
                self.timeEyesClosed += self.timeDiffInSec
                drowsData['ETC'] = self.timeEyesClosed                        
                self.calc_eyes_closed_stats() # User has opened the eyes it is time now to sum up statistics for when eyes were closed
            elif (self.eyesClosed == False): # User has opened eyes and had them opened before
                timeNow = dt.datetime.now()  # Get current time
                self.timeDiffInSec = (timeNow - self.timeStartEyesOpen).microseconds / 1e6  # Increase the time eyes have been opened
                self.timeStartEyesOpen = timeNow
                drowsData['ETO'] += self.timeDiffInSec   # Provide the time eyes have been opened
                drowsData['ETC'] = 0.0  # Eyes are not closed so set that time to zero
                self.timeEyesClosed = 0.0
        
        # Provide drows data back
        # Determine and provide drows level
        # ETO and ETC are set earlier in the function
        drowsData['DLEVEL'] = self.determine_drows_level()
        drowsData['EAR'] = ear
        drowsData['BPM'] = self.currentBlinkCounter
        drowsData['MAXTCPM'] = self.currentMaxEyesClosed
        drowsData['MINTCPM'] = self.currentMinEyesClosed
        drowsData['ATCPM'] = self.currentSumEyesClosed

        # Find if one minute sample was completed
        if (self.check_if_min_completed()):
            # Minute completed so we want to calculate deltas
            drowsData['DBPM'] = self.currentBlinkCounter - self.prevBlinkCounter
            drowsData['DATCPM'] = self.currentSumEyesClosed - self.prevSumEyesClosed
            # Save current data as previous data and zero out the stats
            self.prevBlinkCounter = self.currentBlinkCounter
            self.currentBlinkcounter = 0.0
            self.prevSumEyesClosed = self.currentSumEyesClosed
            self.currentSumEyesClosed = 0.0
            self.prevBlinkCounter - self.currentBlinkCounter
            self.currentBlinkCounter = 0.0
            self.prevMaxEyesClosed = self.currentMaxEyesClosed
            self.currentMaxEyesClosed = 0.0
            self.prevMinEyesClosed = self.currentMinEyesClosed
            self.currentMinEyesClosed = 0.0
            # Save current time to restart counting another minute
            self.oneMinuteTimerStart = dt.datetime.now()
            self.minuteCompleted = False

        # Return the image
        return frame        

    def check_if_min_completed(self):
        timeNow = dt.datetime.now()
        timeDiffInSec = (timeNow - self.oneMinuteTimerStart).microseconds / 1e6
        self.oneMinuteTimerStart = timeNow
        self.minuteCounter += timeDiffInSec 
        if (self.minuteCounter >= 60.0):
            self.minuteCompleted = True
            self.minuteCounter = 0.0
        else:
            self.minuteCompleted = False    

        return self.minuteCompleted
    
    def check_if_user_asleep(self, timeDiffInSec):
        if (timeDiffInSec >= self.ASLEEP_THRESH):
            self.isUserAsleep = True
        else:
            self.isUserAsleep = False


    def determine_drows_level(self):
        if (self.isUserAsleep):
            return 'EXTREME'
        else:
            # For now I will return LOW in any other case but in the end
            # Here we need to have an algorithm that determines the drows level
            return 'LOW'

    def calc_eyes_closed_stats(self):
        self.currentSumEyesClosed += self.timeEyesClosed
        # Determine if the time eyes were closed now was max or min
        if (self.timeEyesClosed > self.currentMaxEyesClosed):
            self.currentMaxEyesClosed = self.timeEyesClosed
        elif ((self.timeEyesClosed < self.currentMinEyesClosed and self.timeEyesClosed != 0.0)
              or self.currentMinEyesClosed == 0.0):
            self.currentMinEyesClosed = self.timeEyesClosed

class SleepDetectorApp: 
    def __init__(self, tk_window, args):
        self.window = tk_window 
        # Set up variables used by the class
        # Map to convert drows level to color display
        self.drowsLevelToDispColors = {'LOW' : 'green', 'MEDIUM' : 'yellow',
                                        'HIGH' : '#995C00', 'EXTREME' : 'red'}
        # Map used by all layers of the app: Model, View and  Controller to work with data
        self.drowsData = {'EAR' : 0.0, 'ETC' : 0.0, 'ETO' : 0.0, 'BPM' : 0.0, 'DBPM' : 0.0,
                              'ATCPM' : 0.0, 'DATCPM' : 0.0, 'MAXTCPM' : 0.0, 'MINTCPM' : 0.0, 'DLEVEL' : 'LOW'} 
        # Delay in miliseconds for how often image frame should be grabbed and calc performed
        self.delay = 5
        # Create video stream for getting images (controller)
        self.videoStream = VideoStream().start()
        # Create instance of drows utils used to analyze data
        self.drowsAnalyst = DrowsAnalyst(args, self.delay)
        # Create instance of warning announcer used to output warning
        self.warnAnnouncer = WarningAnnouncer(args)
        # Give camera sensor time to warm up and music to be loaded
        time.sleep(1.0)
        # Create the view
        self.create_view()
        # Call update function to start getting video frames
        self.update()
        # Start the app main loop
        self.window.mainloop()

    def create_view(self):
        # Set up GUI elements    
        # Create main window and make it full screen
        self.window.attributes('-fullscreen', True)
        # Make the application quitable by pressing Escape key
        self.window.bind("<Escape>", quit)
        # Make room for video stream by creating a canvas
        self.videoOut = tkinter.Canvas(self.window, width = 300, height = 300, bd =4,relief = tkinter.RAISED)
        self.videoOut.place(x = 250, y = 90)
        # Create application title font
        self.titleFont = ('times', 40, 'bold')
        self.title = tkinter.Label(self.window, font = self.titleFont, fg = "black", text = "Sleep Detector")
        self.title.place(x = 240, y = 10)
        # Create string variables that will be used for updating the view
        # Based on the data in the map
        self.drowsLevel = tkinter.StringVar()
        self.drowsLevel.set(self.drowsData['DLEVEL'])
        self.ear = tkinter.StringVar()
        self.ear.set('{:.4f}'.format(self.drowsData['EAR']))
        self.etc = tkinter.StringVar()
        self.etc.set('{:.4f}'.format(self.drowsData['ETC']))
        self.eto = tkinter.StringVar()
        self.eto.set('{:.4f}'.format(self.drowsData['ETO']))
        self.bpm = tkinter.StringVar()
        self.bpm.set('{:.4f}'.format(self.drowsData['BPM']))
        self.dbpm = tkinter.StringVar()
        self.dbpm.set('{:.4f}'.format(self.drowsData['DBPM']))
        self.atcpm = tkinter.StringVar()
        self.atcpm.set('{:.4f}'.format(self.drowsData['ATCPM']))
        self.datcpm = tkinter.StringVar()
        self.datcpm.set('{:.4f}'.format(self.drowsData['DATCPM']))
        self.maxtcpm = tkinter.StringVar()
        self.maxtcpm.set('{:.4f}'.format(self.drowsData['MAXTCPM']))
        self.mintcpm = tkinter.StringVar()
        self.mintcpm.set('{:.4f}'.format(self.drowsData['MINTCPM']))
        # Create a bar on the bottom to display drowsiness level
        self.drowsLevelFrame = tkinter.Frame(self.window, relief = tkinter.RAISED, bg = self.drowsLevelToDispColors['LOW'], width = 800 , height = 80)
        self.drowsLevelFrame.place(x = 0, y = 400)
        self.drowsLevelFont = ('times', 20, 'bold')
        self.drowsLevelTxt = tkinter.Label(self.drowsLevelFrame, font = self.drowsLevelFont, bg = self.drowsLevelToDispColors['LOW'] , textvariable = self.drowsLevel, width = 10 , height = 1)
        self.drowsLevelTxt.place(x = 340, y = 20) 
        # Creat labels for real time data and statistics
        # Real time data
        self.regTextFont = ('times', 15, 'bold')
        self.realTimeLabel = tkinter.Label(self.window, font = self.regTextFont, text = "Real time data", width = 12)
        self.realTimeLabel.place(x = 60, y = 90)
        self.statsLabel = tkinter.Label(self.window, font = self.regTextFont, text = "Statistics", width = 12)
        self.statsLabel.place(x = 610, y = 90)
        self.dataTextFont = ('times', 15)
        # Eye aspect ratio
        self.earLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "EAR")
        self.earLabel.place(x = 10, y = 150)
        self.earValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.ear, anchor = tkinter.W, width = 6)
        self.earValue.place(x = 120, y = 150)
        # Eyes time closed
        self.etcLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "ETC")
        self.etcLabel.place(x = 10, y = 190)
        self.etcValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.etc, anchor = tkinter.W, width = 6)
        self.etcValue.place(x = 120, y = 190)
        self.etoLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "ETO")
        # Eyes time open
        self.etoLabel.place(x = 10, y = 230)
        self.etoValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.eto, anchor = tkinter.W, width = 6)
        self.etoValue.place(x = 120, y = 230)
        # Statistics data
        # Blinks per minute
        self.bpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "BPM")
        self.bpmLabel.place(x = 560, y = 150)
        self.bpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.bpm, anchor = tkinter.W, width = 6)
        self.bpmValue.place(x = 670, y = 150)
        # Delta blinks per minute -> current - previous
        self.dbpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "DBPM")
        self.dbpmLabel.place(x = 560, y = 190)
        self.dbpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.dbpm, anchor = tkinter.W, width = 6)
        self.dbpmValue.place(x = 670, y = 190)
        # Average time closed during blink in last minute
        self.atcpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "ATCPM")
        self.atcpmLabel.place(x = 560, y = 230)
        self.atcpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.atcpm, anchor = tkinter.W, width = 6)
        self.atcpmValue.place(x = 670, y = 230)
        # Delta average time closed during blink -> current - previous
        self.datcpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "DATCPM")
        self.datcpmLabel.place(x = 560, y = 270)
        self.datcpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.datcpm, anchor = tkinter.W, width = 6)
        self.datcpmValue.place(x = 670, y = 270)
        # Longest time eyes were closed in last minute
        self.maxtcpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "MAXTCPM")
        self.maxtcpmLabel.place(x = 560, y = 310)
        self.maxtcpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.maxtcpm, anchor = tkinter.W, width = 6)
        self.maxtcpmValue.place(x = 670, y = 310)
        # Min time eyes were closed in last minute
        self.mintcpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "MINTCPM")
        self.mintcpmLabel.place(x = 560, y = 350)
        self.mintcpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.mintcpm, anchor = tkinter.W, width = 6)
        self.mintcpmValue.place(x = 670, y = 350)

    def update_view(self):
        self.drowsLevelFrame.config(bg = self.drowsLevelToDispColors[self.drowsData['DLEVEL']])
        self.drowsLevelTxt.config(bg = self.drowsLevelToDispColors[self.drowsData['DLEVEL']])
        self.drowsLevel.set(self.drowsData['DLEVEL'])
        self.ear.set('{:.4f}'.format(self.drowsData['EAR']))
        self.etc.set('{:.4f}'.format(self.drowsData['ETC']))
        self.eto.set('{:.4f}'.format(self.drowsData['ETO']))
        self.bpm.set('{:.4f}'.format(self.drowsData['BPM']))
        self.dbpm.set('{:.4f}'.format(self.drowsData['DBPM']))
        self.atcpm.set('{:.4f}'.format(self.drowsData['ATCPM']))
        self.datcpm.set('{:.4f}'.format(self.drowsData['DATCPM']))
        self.maxtcpm.set('{:.4f}'.format(self.drowsData['MAXTCPM']))
        self.mintcpm.set('{:.4f}'.format(self.drowsData['MINTCPM']))

    def update(self):
        frame = self.videoStream.read()
        frame = self.drowsAnalyst.provide_drows_data(frame, self.drowsData)
        self.update_view()
        
        # Determine if we need to trigger the warning
        if (self.drowsData['DLEVEL'] == 'EXTREME'):
            self.warnAnnouncer.warn()
        else:
            self.warnAnnouncer.stop_warning()
        
        image = PIL.Image.fromarray(frame)
        image = image.resize((300,300))
        self.photo = PIL.ImageTk.PhotoImage(image)
        self.videoOut.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        # Make sure we keep getting the frames every 15 ms
        self.window.after(self.delay, self.update)


# Use argument parses to get paths for Haas cascade XML and shape predictor
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascade", required = True, help = "path to Haar cascade XML")
parser.add_argument("-p", "--shape-predictor", required = True, help = "path to facial landmark predictor data")
parser.add_argument("-w", "--warning-sound", required = False, help = "path to warning sound")
# Parsed arguments
args = vars(parser.parse_args())

# Create a window and pass it to Sleep detector application
# Also pass in a list of terminal arguments
SleepDetectorApp(tkinter.Tk(), args)
        