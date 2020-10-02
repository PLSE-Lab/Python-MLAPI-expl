# mathetmatical functions
import numpy as np
# outlier removal function using interquartile range
def reject_outliers_iqr(data, metric):
    metric_formatted = data[metric].astype(float)
    print(metric_formatted)
    q1, q3 = np.percentile(metric_formatted, [25, 75])
    iqr = q3 - q1
    
    lower_bound = q1 - (iqr * 1.5)
    upper_bound = q3 + (iqr * 1.5)
    
    data = data[(metric_formatted > lower_bound) & (metric_formatted < upper_bound)]
    
    return data