# -*- coding: utf-8 -*-
import os

"""
Define metadata and paths to be used between files
"""

SAMPLING_RATE = 52 # Data was recorded at 52Hz. 

# File names and locations
DATA_DIR = os.path.join(os.getcwd(), 'data')
CSV_FILES = [os.path.join(DATA_DIR, f) for f 
        in os.listdir(DATA_DIR) 
        if f.endswith('.csv')]

# Column names
DATA_COLS = ["x_accel", "y_accel", "z_accel"] # time series data
TARGET_COL = ["target"] # the activity
COLS = ["_idx"] + DATA_COLS + TARGET_COL

TARGET_DEFS = {
    1: "Working at Computer",
    2: "Standing Up, Walking and Going updown stairs",
    3: "Standing",
    4: "Walking",
    5: "Going UpDown Stairs",
    6: "Walking and Talking with Someone",
    7: "Talking while Standing",        
}
VALID_TARGETS = list(TARGET_DEFS.keys()) # i.e: 1-7