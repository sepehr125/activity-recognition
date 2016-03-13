import numpy as np

####################
### Helper functions
####################
def window(width, overlap, max_idx):
    """
    Generates tuples of indices that define a window
    of given width and overlap. 
    
    For example:
    window(width=10, overlap=0.5, max_length=30)
    (0, 10)
    (5, 15)
    (10, 20)
    (15, 25)
    Note: it trims the end; i.e. won't return (25, 30)
    """
    start = 0
    if overlap < 0.0 or overlap >= 1.:
        raise ValueError("overlap needs to be a number between 0 and 1")
    while True:
        end = start + width
        if end >= max_idx:
            return None
        yield start, end
        start += max(int((1-overlap)*width), 1)

        
def window_df(df, width, overlap):
    """
    Applies window to a dataframe to return chunks of rows,
    with overlap if specified.
    """
    windows = window(width, overlap, len(df))
    for start, end in windows:
        yield df[start:end]


def standardize(df):
    """
    Make the mean of data 0 and normalize
    """
    return (df - df.mean()) / df.std()

def top_k_indices(series, k):
    return np.argsort(series)[::-1][0:k]