def find_outliers_tukey(x, top = 75, bottom = 25):
    """
    Outlier detection
    Input: column of a df, optional top and bottom thresholds as percentiles
        - 
    Output: two lists, one of indices, one of values
        - 
    """
    import numpy as np
    q1 = np.percentile(x, bottom)
    q3 = np.percentile(x, top)
    iqr = q3-q1
    floor = q1- 1.5 * iqr
    ceiling = q3 + 1.5 * iqr
    outlier_indices = list(x.index[(x< floor)|(x> ceiling)])
    outlier_values = list(x[outlier_indices])
    
    return outlier_indices, outlier_values