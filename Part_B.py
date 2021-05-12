# -*- coding: utf-8 -*-
"""
Created on Sun May  2 01:15:19 2021

@author: art27

Part_B
A lot of this is recycled code from Lab 5
"""

# %% Imports

import numpy as np
import sklearn as skl # The ML module
from sklearn import preprocessing as pre # I dont know why but if I don't I get an 
from sklearn import model_selection, svm # really sklearn, really? Dot notation, ever heard of it?
from joblib import dump

# %% Parser Changes //If any

# %% Load in data // Recyled

def load_data(name): # Time removed for now as test sets do not include //TODO generalize to check for existance of time file?
    '''load_data
    Parameters-
    name : str : The name associated with the loaded files
    Returns
    emg_data : 2D np array : the emg sensor data
    emg_time : 1D np array : the timestamps of associated samples
    
    loads both emg and time data of specified name
    '''
    emg_data = np.load(str(name) + '_RPS_Data.npy') # Load in data
    #emg_time = np.load(str(name) + '_RPS_Time.npy') # Load in time
    
    return emg_data #, emg_time # Return loaded data and time

# %% Process and Extract data // Recyled

def epoch_data(emg_data, fs, epoch_duration):
    '''epoch_data
    Parameters
    emg_data : 2D np array : the emg sensor data
    fs : int : the sampling frequency
    epoch_duration : float : the length of each epoch in seconds
    Returns
    epoched_data : 3D np array : the emg data divided into epochs
    
    Seperates the given emg data into epochs
    '''
    # Subtract the mean offset from each channel
    for channel in range(np.size(emg_data, 1)): # Double check the array dimensions
        mean = np.mean(emg_data[:,channel]) # Find the mean of the current channel
        emg_data[:,channel] -= mean # Subtract the mean across the current channel
        #--- Fun fact, -= across an array only works for  numpy arrays, not on lists
    
    # epoch_count defined as variable for readability
    epoch_count = int((np.size(emg_data, 0) / (fs*epoch_duration)) + 0.5) # int and + 0.5 to account for uneven divsion
    # Initialize epoched_data array
    #epoched_data = np.zeros(epoch_count, fs*epoch_duration, np.size(emg_data, 1))
    epoched_data = np.reshape(emg_data, (epoch_count,fs*epoch_duration,np.size(emg_data, 1)))
    
    return epoched_data # Return the epoched_data array

def extract_features(epoched_data):
    '''extract_features
    Parameters
    epoched_data : 3D np array : the emg data split into epochs
    Returns
    features : 2D np array : several calculated parameters of the epoched data
    feature_shorthands : 1D str list : a list of the feature name shorthands
    
    Exctracts and calculates a set of features for use in ML // Variance, Mean Absolute Vale (MAV), Zero Crossing events (ZC)
    '''
    # Initialie the feature sub-matrices
    epoch_var = np.zeros([np.size(epoched_data, 0), np.size(epoched_data, 2)]) # variance
    epoch_mav = np.zeros([np.size(epoched_data, 0), np.size(epoched_data, 2)]) # mean absolute value
    epoch_zc = np.zeros([np.size(epoched_data, 0), np.size(epoched_data, 2)]) # number of zero crossing events
    
    for epoch in range(np.size(epoched_data, 0)): # iterate through epochs
        for channel in range(np.size(epoched_data, 2)): # iterates through channels
            # Set the values of the variance and MAV sub-matrices
            epoch_var[epoch, channel] = np.nanvar(epoched_data[epoch, :, channel])
            epoch_mav[epoch, channel] = np.mean(np.abs(epoched_data[epoch, :, channel]))
            
            zc = 0 # zero crossing count
            prev_sign = np.sign(epoched_data[epoch,0,channel]) # Pulls the sign of the first sample
            for sample in range(np.size(epoched_data, 1)): # Iterates through all the samples in the current epoch and channel
                if np.sign(epoched_data[epoch,sample,channel]) != prev_sign: # Check if the sign of the current value is different than previous sign
                    zc += 1 # Increment zc when sign is different
                prev_sign = np.sign(epoched_data[epoch,sample,channel])
            epoch_zc[epoch, channel] = zc # set the epoch and channel zero crossing count to zc
        
    # Concatenate the submatrices column-wise into the full feature array
    features = np.concatenate((epoch_var,epoch_mav,epoch_zc),axis=1)
    
    # Normalize the feature array
    features = pre.scale(features)
    #--- Cannot get mean and std exactly 0 and 1. Tested on smaller arrays with same syntax and was succesful
    #--- Output mean and std where X*10^-17 and 0.99999... respectively, assuming floating point error or issue with big array
    
    # Create the shorthand list
    #feature_shorthands = ['Var_ch0','Var_ch1','Var_ch2','MAV_ch0','MAV_ch1','MAV_ch2','ZC_ch0','ZC_ch1','ZC_ch2'] # Why is this even in this method?
    
    return features #, feature_shorthands # Return the normalized feature array

def make_truth_data(action_sequence, epochs_per_action):
    '''make_truth_data
    Parameters
    action_sequence : 1D str list : the whole seuqence of actions over the recording period
    epochs_per_action : int : the number of epochs per second
    Returns
    instructed_action : 1D str list : the total list of actions acounting for duplicate actions each second due to multipole epochs per second
    
    Creates a list of true actions factoring in repeats due to multiple epochs per second
    '''
    instructed_action = [] # Create the empty list for instructed actions
    for action in range(len(action_sequence)): # I guess we're assuming non-fractional values at this point? (I'd code around that but time)
        for epoch in range(epochs_per_action): # This could probably be done more elegantly but...
            instructed_action.append(action_sequence[action]) # Append the current action epochs_per_action times
        
    return instructed_action # Returns the array of instructed actions at associated epochs

# %% Train and validate classifier

def create_train_classifier(data, targets, param_grid={'C':[[0.1,1,10,100,1e6]],'kernel':[['poly','rbf']]}): # //TODO Add default values
    '''create_train_classifier
    Parameters
    data : np array (or array-like) : the data to be split and train the classifier, has a default
    param_grid : dictionary : set of parameters to test
    Returns
    
    '''
    f_train, f_test, a_train, a_test = skl.model_selection.train_test_split(data, targets) # split the given data into test and training sets
    svc = svm.SVC(decision_function_shape='ovr') # Create the classifier object
    search = model_selection.GridSearchCV(svc, skl.model_selection.ParameterGrid(param_grid)) # Run the grid search, using ParameterGrid to permute the parameters
    search.fit(f_train, a_train) # Fit the data set
    
    dump(search, 'classifier.joblib') # Save the classifier object
    
    return search

# %% Main calls and testing

#['rest','rock','rest','paper','rest','scissors']