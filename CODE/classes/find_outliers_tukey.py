def find_outliers_tukey(x):
    """
    Outlier detection
    Input:
        - 
    Output:
        - 
    """
    q1 = np.percentile(x,25)
    q3 = np.percentile(x,75)
    iqr = q3-q1
    floor = q1- 1.5 * iqr
    ceiling = q3 + 1.5 * iqr
    outlier_indices = list(x.index[(x< floor)|(x> ceiling)])
    outlier_values = list(x[outlier_indices])
    
    return outlier_indices, outlier_values