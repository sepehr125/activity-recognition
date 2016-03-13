# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict
import argparse
from helpers import window_df, standardize, amplitudes
from settings import SAMPLING_RATE, CSV_FILES, VALID_TARGETS, \
                        DATA_COLS, TARGET_COL, COLS

"""
This script:
1. Reads CSV files from globally defined location in settings.py
2. Windows over data to compute aggregate features. 
3. Saves feature matrix (pandas dataframe) as a pickled file.
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                        description=
                        'Window over data to compute \
                        aggregate features. Save results')

    # Window width = n_seconds * SAMPLING_RATE
    parser.add_argument(
                        'n_seconds',
                        help="Window duration",
                        default=5, type=int)

    # Windows move forward in increments of pct_overlap * window width
    parser.add_argument(
                        'pct_overlap', 
                        help="[0,1) Proportion of window width \
                            that should overlap in consecutive samples.",
                        default=0.5, type=float)

    parser.add_argument(
                        'output_file',
                        default="feature_matrix.pkl"
                        help="Path or name of file to pickle results.")
    args = parser.parse_args()


    # usecols param lets us read in specific columns only. 
    # Ignore index column because we re-index to ensure integrity.
    dfs = [pd.read_csv(f, names=COLS, usecols=DATA_COLS+TARGET_COL) for f in CSV_FILES]
    
    # Concatenate all files into one big dataframe
    master = pd.concat(dfs).reset_index()

    # Ensure targets are valid values (1-7). There are some 0s in there.
    master_valid = master[master['target'].isin(VALID_TARGETS)]

    feature_matrix = []
    for target, df in master_valid.groupby('target'):
        
        # 0-center the mean and normalize. this adds about 5 points
        df[DATA_COLS] = standardize(df[DATA_COLS])
        
        new_group = defaultdict(list)
        new_group['target'] = target
        samples = window_df(df, args.n_seconds*SAMPLING_RATE, args.pct_overlap)
        for sample in samples:
            
            means = sample[DATA_COLS].mean()
            new_group['x_accel_mean'].append(means['x_accel'])
            new_group['y_accel_mean'].append(means['y_accel'])
            new_group['z_accel_mean'].append(means['z_accel'])
            
            stds = sample[DATA_COLS].std()
            new_group['x_accel_std'].append(stds['x_accel'])
            new_group['y_accel_std'].append(stds['y_accel'])
            new_group['z_accel_std'].append(stds['z_accel'])
            
            new_group['x_max_min'].append(max(sample.x_accel) - min(sample.x_accel))
            new_group['y_max_min'].append(max(sample.y_accel) - min(sample.y_accel))
            new_group['z_max_min'].append(max(sample.z_accel) - min(sample.z_accel))
            
            # correlations
            corrs = sample[DATA_COLS].corr()
            new_group['xy_corr'].append(corrs.loc['x_accel', 'y_accel'])
            new_group['xz_corr'].append(corrs.loc['x_accel', 'z_accel'])
            new_group['yz_corr'].append(corrs.loc['y_accel', 'z_accel'])
            
            # root-mean-square(x, y, z)
            rms = np.sqrt(np.mean(np.square(sample[DATA_COLS]), axis=1))
            new_group['rms_mean'].append(rms.mean())
            new_group['rms_std'].append(rms.std())

            
            # fourier transforms! 
            x_fft = amplitudes(sample['x_accel'])
            y_fft = amplitudes(sample['y_accel'])
            z_fft = amplitudes(sample['z_accel'])
           
            # Max Fourier 
            new_group['x_fft_max'].append(x_fft.max())
            new_group['y_fft_max'].append(y_fft.max())
            new_group['z_fft_max'].append(z_fft.max())

            # Min Fourier
            new_group['x_fft_min'].append(x_fft.min())
            new_group['y_fft_min'].append(y_fft.min())
            new_group['z_fft_min'].append(z_fft.min())
            
            # Mean Fourier
            new_group['x_fft_mean'].append(x_fft.mean())
            new_group['y_fft_mean'].append(y_fft.mean())
            new_group['z_fft_mean'].append(z_fft.mean())

            # Standard deviation Fourier
            new_group['x_fft_std'].append(x_fft.std())
            new_group['y_fft_std'].append(y_fft.std())
            new_group['z_fft_std'].append(z_fft.std())


        # Add new_group to feature_matrix
        feature_matrix.append(pd.DataFrame(new_group))
        print("Finished target #%d"%target)

    # concatenate all groups into one dataframe
    feature_matrix_df = pd.concat(feature_matrix)


    with open(args.output_file, 'wb') as f:
        pickle.dump(feature_matrix_df, f)