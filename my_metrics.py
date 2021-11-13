"""H-measure and KS score"""
import numpy as np
import hmeasure
import scipy.stats

def h_socre(y_true, y_pred, **kwargs):
    """H-measure"""
    class_true = np.asarray(y_true)
    result = hmeasure.h_score(class_true, y_pred, **kwargs)
    return result

def ks_socre(y_true, y_pred, **kwargs):
    """Kolmogorov–Smirnov statistic"""
    result = scipy.stats.ks_2samp  \
        (y_pred[y_true==1], y_pred[y_true!=1], **kwargs).statistic
    return result

if __name__ == "__main__":
    pass
