# -*- coding: utf-8 -*-
"""
Created on Wed May 12 00:10:28 2021

@author: Artur Smiechowski
"""

# %% Imports
from Part_B import load_data, epoch_data, extract_features, make_truth_data, create_train_classifier
import numpy as np
import sklearn as skl # The ML module
from sklearn import preprocessing as pre # I dont know why but if I don't I get an 
from sklearn import model_selection, svm # really sklearn, really? Dot notation, ever heard of it?

# %% Data loading (kept global to tinker in console)

# Data sets A-E loaded in and returned as feature arrays via method cascade
A = extract_features(epoch_data(load_data("A"), 500, 1))
B = extract_features(epoch_data(load_data("B"), 500, 1))
C = extract_features(epoch_data(load_data("C"), 500, 1))
D = extract_features(epoch_data(load_data("D"), 500, 1))
E = extract_features(epoch_data(load_data("E"), 500, 1))

# %% Grid Search and training

