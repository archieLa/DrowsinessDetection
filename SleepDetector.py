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
            # Map to convert drows level to color display
            self.drowsLevelToDispColors = {'LOW' : 'green', 'MEDIUM' : 'yellow',
                                           'HIGH' : '#995C00', 'EXTREME' : 'red'}
            # Map used by all layers of the app: Model, View and  Controller to work with data
            self.drowsData = {'EAR' : 0.0, 'ETC' : 0.0, 'ETO' : 0.0, 'BPM' : 0.0, 'DBPM' : 0.0,
                              'ATCPM' : 0.0, 'DATCPM' : 0.0, 'MAXTCPM' : 0.0, 'MINTCPM' : 0.0, 'DLEVEL' : 'LOW'} 
            # Create GUI and start the application
            self.create_view()
            self.window.mainloop()

        def create_view(self):
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
            # Create string variables that will be used for updating the view
            # Based on the data in the map
            self.drowsLevel = tkinter.StringVar()
            self.drowsLevel.set(self.drowsData['DLEVEL'])
            self.ear = tkinter.StringVar()
            self.ear.set('{:.2f}'.format(self.drowsData['EAR']))
            self.etc = tkinter.StringVar()
            self.etc.set('{:.2f}'.format(self.drowsData['ETC']))
            self.eto = tkinter.StringVar()
            self.eto.set('{:.2f}'.format(self.drowsData['ETO']))
            self.bpm = tkinter.StringVar()
            self.bpm.set('{:.2f}'.format(self.drowsData['BPM']))
            self.dbpm = tkinter.StringVar()
            self.dbpm.set('{:.2f}'.format(self.drowsData['DBPM']))
            self.atcpm = tkinter.StringVar()
            self.atcpm.set('{:.2f}'.format(self.drowsData['ATCPM']))
            self.datcpm = tkinter.StringVar()
            self.datcpm.set('{:.2f}'.format(self.drowsData['DATCPM']))
            self.maxtcpm = tkinter.StringVar()
            self.maxtcpm.set('{:.2f}'.format(self.drowsData['MAXTCPM']))
            self.mintcpm = tkinter.StringVar()
            self.mintcpm.set('{:.2f}'.format(self.drowsData['MINTCPM']))
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
            self.earValue.place(x = 160, y = 150)
            # Eyes time closed
            self.etcLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "ETC")
            self.etcLabel.place(x = 10, y = 190)
            self.etcValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.etc, anchor = tkinter.W, width = 6)
            self.etcValue.place(x = 160, y = 190)
            self.etoLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "ETO")
            # Eyes time open
            self.etoLabel.place(x = 10, y = 230)
            self.etoValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.eto, anchor = tkinter.W, width = 6)
            self.etoValue.place(x = 160, y = 230)
            # Statistics data
            # Blinks per minute
            self.bpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "BPM")
            self.bpmLabel.place(x = 560, y = 150)
            self.bpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.bpm, anchor = tkinter.W, width = 6)
            self.bpmValue.place(x = 710, y = 150)
            # Delta blinks per minute -> current - previous
            self.dbpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "DBPM")
            self.dbpmLabel.place(x = 560, y = 190)
            self.dbpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.dbpm, anchor = tkinter.W, width = 6)
            self.dbpmValue.place(x = 710, y = 190)
            # Average time closed during blink in last minute
            self.atcpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "ATCPM")
            self.atcpmLabel.place(x = 560, y = 230)
            self.atcpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.atcpm, anchor = tkinter.W, width = 6)
            self.atcpmValue.place(x = 710, y = 230)
            # Delta average time closed during blink -> current - previous
            self.datcpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "DATCPM")
            self.datcpmLabel.place(x = 560, y = 270)
            self.datcpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.datcpm, anchor = tkinter.W, width = 6)
            self.datcpmValue.place(x = 710, y = 270)
            # Longest time eyes were closed in last minute
            self.maxtcpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "MAXTCPM")
            self.maxtcpmLabel.place(x = 560, y = 310)
            self.maxtcpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.maxtcpm, anchor = tkinter.W, width = 6)
            self.maxtcpmValue.place(x = 710, y = 310)
            # Min time eyes were closed in last minute
            self.mintcpmLabel = tkinter.Label(self.window, font = self.dataTextFont, text = "MINTCPM")
            self.mintcpmLabel.place(x = 560, y = 350)
            self.mintcpmValue = tkinter.Label(self.window, font = self.dataTextFont, textvariable = self.mintcpm, anchor = tkinter.W, width = 6)
            self.mintcpmValue.place(x = 710, y = 350)

        def update_view(self):
            self.drowsLevelFrame.config(bg = self.drowsLevelToDispColors[self.drowsData['DLEVEL']])
            self.drowsLevelTxt.config(bg = self.drowsLevelToDispColors[self.drowsData['DLEVEL']])
            self.drowsLevel.set(self.drowsData['DLEVEL'])
            self.ear.set('{:.2f}'.format(self.drowsData['EAR']))
            self.etc.set('{:.2f}'.format(self.drowsData['ETC']))
            self.eto.set('{:.2f}'.format(self.drowsData['ETO']))
            self.bpm.set('{:.2f}'.format(self.drowsData['BPM']))
            self.dbpm.set('{:.2f}'.format(self.drowsData['DBPM']))
            self.atcpm.set('{:.2f}'.format(self.drowsData['ATCPM']))
            self.datcpm.set('{:.2f}'.format(self.drowsData['DATCPM']))
            self.maxtcpm.set('{:.2f}'.format(self.drowsData['MAXTCPM']))
            self.mintcpm.set('{:.2f}'.format(self.drowsData['MINTCPM']))


# Create a window and pass it to Sleep detector application
SleepDetectorApp(tkinter.Tk())
        