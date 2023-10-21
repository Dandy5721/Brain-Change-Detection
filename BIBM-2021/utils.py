import numpy as np
import re
from numpy.lib.stride_tricks import sliding_window_view
from sklearn import metrics
from sklearn.metrics import accuracy_score, normalized_mutual_info_score


def corrcoef(X):
    avg = X.mean(-1)
    X = X - avg[..., None]
    X_T = X.swapaxes(-2, -1)
    c = X @ X_T
    d = c.diagonal(0, -2, -1)
    stddev = np.sqrt(d)
    c /= stddev[..., None]
    c /= stddev[..., None, :]
    np.clip(c, -1, 1, out=c)
    return c


def sliding_window_corrcoef(X, window, padding=True):
    if padding:
        left = (window - 1) // 2
        right = window - 1 - left
        X = np.concatenate((X[..., :left], X, X[..., -right:]), axis=-1)
    X_window = sliding_window_view(X, (X.shape[-2], window), (-2, -1)).squeeze()
    return corrcoef(X_window)


def purity_score(y_true, y_pred):
    # compute contingency matrix
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


def cluster_score(labels_true, labels_pred):
    purity = purity_score(labels_true, labels_pred)
    acc = accuracy_score(labels_true, labels_pred)
    nmi = normalized_mutual_info_score(labels_true, labels_pred)
    return purity, acc, nmi


def sorted_aphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
