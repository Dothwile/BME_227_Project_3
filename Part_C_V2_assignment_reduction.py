#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:30:55 2021

Project 3 Part C V.1

@author: Ian
"""
# %% Imports
# import packages
import pickle
import numpy as np
import argparse
import serial
import pyautogui as pyt
from Part_B import extract_features, epoch_data
from joblib import load
# turn off the corner interrupt 
pyt.FAILSAFE = False

# %% CONSTANTS

# global Definitions
BAUD_RATE = 500000 # arduino baudrate (kps)
PORT_LOCATION = '/dev/cu.usbserial-1410' # arduino port
UPPER_VOLTAGE = 5 # arduino is 5V max
SIGNAL_BITS = 1024 # bits for signal to voltage conversion
MS_TO_SEC = 1000.0 # convert ms to s
MAX_FAIL_ATTEMPS = 3 # num failed connections (attemps)
EPOCH_DURATION =  1


# %% initialize arrays

def initialize_arrays(recording_duration, n_channels, fs):
    """initialize_arrays creates and return a data and time array
    
    Keyword Arguments:
    recording_duration -- duration of sample in seconds
    n_channels --number of channels in data stream
    fs -- sampling frequency of data
    
    Return:
    [sample_data, sample_time] -- 2 arrays of data and time information
    """

    # initialize sample_data 2D array, length based on time value
    # rows are samples, columns are channels
    sample_data = np.zeros((recording_duration*fs,n_channels), dtype=float)*np.nan

    # initialize sample_time 1D array, length beased on time value
    # each row is anticipated time of sample. 
    sample_time = np.arange(recording_duration*fs, dtype=float)

    # return sample data and time arrays
    return [sample_data, sample_time]

# main function to read and plot data
def predict_and_move(com_port, recording_duration, fs, epoch_length, classifier_file, slowness):
    
    """Main Function to read and plot data in real time
        Data and plot is saved as a ".npy" array

    Keyword Arguments:
    com_port -- port location for Arduino
    recording_duration -- sample time in seconds
    n_channels -- number of channels
    """
    # program written to work with 3 channels
    n_channels = 3
    
    # figure out how many samples are used to make a decision
    samples_per_epoch = int((epoch_length/1000)*fs) 

    # read in threshold array from external source
    
    # Load in the classifier object to live label actions
    classifier = load('classifier.joblib')
    
    # Determine screen resolution and resulting button locations
    x_max, y_max = pyt.size() # Maximum x and y values of the screen (resolution dependent)
    rock_pos = (x_max*0.22,y_max*0.55) # Position of rock button
    paper_pos = (x_max*0.33,y_max*0.55) # Position of paper button
    scissors_pos = (x_max*0.44,y_max*0.55) # Position of scissors button

    # connect to arduino
    def open_port():
        '''
        Open the arduino port. Accepts no arguments. Returns the arduino connection.
        Validation if the conneciton takes multiple tries to connect
        Clean exit if the connection process goes wrong
        '''

        # track number of failed connections
        failCount = 0
        arduino = 1

        while (arduino == 1 and failCount < MAX_FAIL_ATTEMPS):
            # connection good
            try:
                arduino = serial.Serial(com_port, baudrate=BAUD_RATE)
            # didn't connect, return error
            except:
                print("Connection Failed, trying again...")
                failCount += 1 # add to fail count
                arduino = 1 # error message
        
        if (arduino != 1):
            return arduino
        else:
            print("Cannot connect to Arduino. Unplug and try again...\nExiting Now")
            #exit()

    def read_plot_data():
        '''
        Read and Plot the data from Serial Monitor. For each expected sample
        read in the line, split accordingly, assign to arrays and plots the data
        in real time (every 100 samples).
        Return an error code
        '''

        # keep track of data read error. If error, exit, try again
        readError = 0
        # read each new sample, add to arrays
        for sample_index in range(1,fs*recording_duration):
            
            try:
                # Read a line from the serial port
                arduino_string = arduino.readline()
                # split elements add to list
                arduino_list = arduino_string.split()

                # extract data from serial monitor and convert
                # time and voltage conversion
                sample_time[sample_index] = float(arduino_list[0]) / MS_TO_SEC
                sample_data[sample_index,:] = np.array(arduino_list[1:n_channels+1], dtype=float) / SIGNAL_BITS * UPPER_VOLTAGE
                
                
                # data is good, no error
                readError = 0

            except: 
                # this means something went wrong, set sample time and data out of bounds
                readError = 1
                return readError            

            # move mouse after each epoch
            if(sample_index % samples_per_epoch == 0 and sample_index != 0):
                epoched_data = epoch_data(sample_data,500,1) # package the data into an epoch
                features = extract_features(epoched_data) # extract the features
                Move_Mouse(classifier, features, slowness) # move the mouse according to classifier

        # everything is good
        return readError # returnError should be '0' at this point
    
    # Move mouse function
    def Move_Mouse(classifier, data_array, mouse_slowness): 
        """
        Move_Mouse parameters are the three threshold values for the three sensors.thresh_1, thres_2, and thresh_3 are the thresholds 
        for the corresponding channel. After these are found in testing they are passed through to the function. The data_array will 
        be a 1x3 array of the three channels and the respective variance ofeach channel.
    
        Debating whether to stick with variance or average of the array input.
    
        """
    
        if classifier.predict(data_array) == 'rest':
            pass
    
        elif classifier.predict(data_array) == 'rock':
            # flex left forearm muscle
            pyt.moveTo(rock_pos, None, mouse_slowness)
            pyt.PAUSE(1)
            pyt.click()
        
        elif classifier.predict(data_array) == 'paper':
            # flex right forearm muscle
            pyt.move(paper_pos, None, mouse_slowness)
            pyt.PAUSE(1)
            pyt.click()
        
        elif classifier.predict(data_array) == 'scissors':
            # flex both forearms
            pyt.moveTo(scissors_pos,None , mouse_slowness)
            pyt.PAUSE(1)
            pyt.click()
        
        else: # On error or errant class ignore
            pass

    # ---- Main Functionality, initialize, read, plot, save ----

    # track fail attempts
    dataAttempts = 0
    readError = 1

    # now read information. If we got this far, conneciton must be good
    while (readError != 0 and dataAttempts < MAX_FAIL_ATTEMPS):
        # Open arduino Connection
        # print message to user
        arduino = open_port()

        # Call array function to get formatted data
        [sample_data, sample_time] = initialize_arrays(recording_duration, n_channels, fs)

        readError = read_plot_data()

        # if there's an error from reading data, close the port and try plotting again
        if (readError != 0):
            # close connection, try again
            arduino.close()
            # add to fail counter
            dataAttempts += 1
    
    # max fail attemps were made, exit gracefully
    if (dataAttempts >= MAX_FAIL_ATTEMPS):
        print("Cannot connect to Arduino. Unplug and try again...\nByeBye")
        arduino.close()
        #exit()
    # if data collection good, save figure
    else:
        arduino.close()

# %% Test

predict_and_move(PORT_LOCATION, 30, 500, 200, 'classifier.joblib', 0.1)
# %%
# call main when running script
if __name__ == '__main__':

    # setup argument parser
    parser = argparse.ArgumentParser(description='Move mouse using muscle contractions') # Create the parser object
    
    parser.add_argument('-p','--port',default=PORT_LOCATION,help='Port for Arduino Connection') # Port location for sensors
    
    parser.add_argument('-d','--duration',default=30,help='Duration of recording in seconds',type=int) # Recording duration in s
    
    parser.add_argument('-fs','--frequency', default=500, help='Sampling frequency in Hz',type=int) # Sample f ion Hz
    
    parser.add_argument("-ep", "--epoch_length_ms", type = int, default = 200,
                    help = "The length of each epoch in milliseconds") # Duration of each sample epoch in ms

    parser.add_argument("-class", "--classifier_file", type = str, default = 
                    'classifier.joblib', help = "The file name of the\
                    classifier produced in setup stage.\
                    Must be located in the same folder as this program.")
                    
    parser.add_argument("-slowness", "--mouse_slowness", type = float, default = 0.1,
                    help = "How long it takes the mouse to move from one\
                    location to another. Smaller inputs lead to faster \
                        mouse movement. Unless you have a very small or\
                        very large monitor, do not enter a value here.") # Inverse of speed value for cursor movement
                        
    args = parser.parse_args() # Collect the parser arguments

    # run main file with arguments from parser
    predict_and_move(args.port, args.duration, args.frequency, args.epoch_length_ms, args.classifier_file, args.mouse_slowness)


