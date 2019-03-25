from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import os
import datetime
import argparse
import time
import dlib
import cv2
import pygame
import tkinter

class SleepDetectorApp:
        def __init__(self, tk_window):
            self.window = tk_window 
            # Set up variables used by the class
            self.drowsLevelToDispColors = {'LOW' : 'green', 'MEDIUM' : 'yellow',
                                           'HIGH' : '#995C00', 'EXTREME' : 'red'}
            self.create_gui()
            self.window.mainloop()

        def create_gui(self):
            # Set up GUI elements    
            # Create main window and make it full screen
            self.window.attributes('-fullscreen', True)
            # Make the application quitable by pressing Escape key
            self.window.bind("<Escape>", quit)
            # Make room for video stream by creating a canvas
            self.videoOut = tkinter.Canvas(self.window, width = 300, height = 300, bd =4, bg = "blue", relief = tkinter.RAISED)
            self.videoOut.place(x = 250, y = 90)
            # Create application title font
            self.titleFont = ('times', 40, 'bold')
            self.title = tkinter.Label(self.window, font = self.titleFont, fg = "black", text = "Sleep Detector")
            self.title.place(x = 240, y = 10)
            # Create a bar on the bottom to display drowsiness level
            self.drowsLevelFrame = tkinter.Frame(self.window, relief = tkinter.RAISED, bg = self.drowsLevelToDispColors['LOW'], width = 800 , height = 80)
            self.drowsLevelFrame.place(x = 0, y = 400)
            self.drowsLevelFont = ('times', 20, 'bold')
            self.drowsLevelTxt = tkinter.Label(self.drowsLevelFrame, font = self.drowsLevelFont, bg = self.drowsLevelToDispColors['LOW'] , text = "LOW", width = 10 , height = 1)
            self.drowsLevelTxt.place(x = 340, y = 20) 
            # Creat labels for real time data and statistics
            self.regTextFont = ('times', 15, 'bold')
            self.realTimeLabel = tkinter.Label(self.window, font = self.regTextFont, text = "Real time data", width = 12)
            self.realTimeLabel.place(x = 60, y = 90)
            self.statsLabel = tkinter.Label(self.window, font = self.regTextFont, text = "Statistics", width = 12)
            self.statsLabel.place(x = 610, y = 90)
            self.dataTextFont = ('times', 15)
            self.statsLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "EAR")
            self.statsLabel.place(x = 10, y = 150)
            self.statsLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "ETC")
            self.statsLabel.place(x = 10, y = 200)
            self.statsLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "ETO")
            self.statsLabel.place(x = 10, y = 250)





# Create a window and pass it to Sleep detector application
SleepDetectorApp(tkinter.Tk())
        