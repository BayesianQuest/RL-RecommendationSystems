'''
This script lists down all the helper functions which are required for processing raw data
'''

from pickle import load
from pickle import dump
import numpy as np


# Function to Save data to pickle form
def save_clean_data(data,filename):
    dump(data,open(filename,'wb'))
    print('Saved: %s' % filename)

# Function to load pickle data from disk
def load_files(filename):
    return load(open(filename,'rb'))






