# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import kurtosis
import os
import pickle
from collections import defaultdict
import argparse
from helpers import window_df, standardize, zero_cross_rate
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

    parser.add_argument(
                        'output_file',
                        default="feature_matrix.pkl",
                        help="Path or name of file to pickle results.")
    
    args = parser.parse_args()

    N_SECONDS = 5
    OVERLAP = 0.5

    # usecols param lets us read in specific columns only. 
    # Ignore index column because we re-index to ensure integrity.
    dfs = [pd.read_csv(f, names=COLS, usecols=DATA_COLS+TARGET_COL) for f in CSV_FILES]
    
    # Concatenate all files into one big dataframe
    master = pd.concat(dfs).reset_index()

    # Ensure targets are valid values (1-7). There are some 0s in there.
    master_valid = master[master['target'].isin(VALID_TARGETS)]

    feature_matrix = []
    # looping over groups helps make sure we don't
    # consider windows with more than one target
    for target, df in master_valid.groupby('target'):
        
        # 0-center the mean and normalize
        df[DATA_COLS] = standardize(df[DATA_COLS])
        
        print("Processing %d rows for target #%d..."%(len(df), target))

        grp = defaultdict(list)
        grp['target'] = target
        samples = window_df(df, 
                            width=N_SECONDS*SAMPLING_RATE, 
                            overlap=OVERLAP)
        for sample in samples:
            
            means = sample[DATA_COLS].mean()
            grp['x_mean'].append(means['x_accel'])
            grp['y_mean'].append(means['y_accel'])
            grp['z_mean'].append(means['z_accel'])
            
            stds = sample[DATA_COLS].std()
            grp['x_std'].append(stds['x_accel'])
            grp['y_std'].append(stds['y_accel'])
            grp['z_std'].append(stds['z_accel'])
            
            grp['x_max_min'].append(max(sample["x_accel"]) - min(sample["x_accel"]))
            grp['y_max_min'].append(max(sample["y_accel"]) - min(sample["y_accel"]))
            grp['z_max_min'].append(max(sample["z_accel"]) - min(sample["z_accel"]))
            
            # correlations
            corrs = sample[DATA_COLS].corr()
            grp['xy_corr'].append(corrs.loc['x_accel', 'y_accel'])
            grp['xz_corr'].append(corrs.loc['x_accel', 'z_accel'])
            grp['yz_corr'].append(corrs.loc['y_accel', 'z_accel'])
            
            # root-mean-square(x, y, z)
            rms = np.sqrt(np.mean(np.square(sample[DATA_COLS]), axis=1))
            grp['rms_mean'].append(rms.mean())
            grp['rms_std'].append(rms.std())

            # zero crossing rate of amplitude (crossing below mean)
            grp['x_zcr'].append(zero_cross_rate(sample['x_accel']))
            grp['y_zcr'].append(zero_cross_rate(sample['y_accel']))
            grp['z_zcr'].append(zero_cross_rate(sample['z_accel']))

            # amplitude kurtosis
            kurtoses = kurtosis(sample[DATA_COLS])
            grp['x_kurtosis'].append(kurtoses[0])
            grp['y_kurtosis'].append(kurtoses[1])
            grp['z_kurtosis'].append(kurtoses[2])

            # fourier transforms!
            x_fft = abs(np.fft.rfft(sample['x_accel']))
            y_fft = abs(np.fft.rfft(sample['y_accel']))
            z_fft = abs(np.fft.rfft(sample['z_accel']))

            grp['x_freq_max'].append(np.argmax(x_fft))
            grp['y_freq_max'].append(np.argmax(y_fft))
            grp['z_freq_max'].append(np.argmax(z_fft))

            # Max Fourier 
            grp['x_fft_max'].append(x_fft.max())
            grp['y_fft_max'].append(y_fft.max())
            grp['z_fft_max'].append(z_fft.max())
            
            # Mean Fourier
            grp['x_fft_mean'].append(x_fft.mean())
            grp['y_fft_mean'].append(y_fft.mean())
            grp['z_fft_mean'].append(z_fft.mean())

            # Standard deviation Fourier
            grp['x_fft_std'].append(x_fft.std())
            grp['y_fft_std'].append(y_fft.std())
            grp['z_fft_std'].append(z_fft.std())

            grp['x_fft_kurtosis'].append(kurtosis(x_fft))
            grp['y_fft_kurtosis'].append(kurtosis(y_fft))
            grp['z_fft_kurtosis'].append(kurtosis(z_fft))

        # Add grp to feature_matrix
        feature_matrix.append(pd.DataFrame(grp))

    # concatenate all groups into one dataframe
    feature_matrix_df = pd.concat(feature_matrix)

    # save features
    with open(args.output_file, 'wb') as f:
        pickle.dump(feature_matrix_df, f)