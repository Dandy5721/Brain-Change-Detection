import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from utils import (
    sliding_window_corrcoef,
    purity_score,
    AverageMeter,
    sorted_aphanumeric,
)
from manifold_mean_shift import ManifoldMeanShift
from tqdm import tqdm
import os

if __name__ == '__main__':

    input_dir = 'data/demo/bolds'
    label_dir = 'data/demo/labels/label.csv'
    window = 30
    max_iter = 200
    n_clusters = 3

    labels_true = np.loadtxt(label_dir, dtype=int)
    purities = AverageMeter()
    nmis = AverageMeter()

    bar = tqdm(sorted_aphanumeric(os.listdir(input_dir)))

    for fname in bar:
        fpath = os.path.join(input_dir, fname)

        bold = np.loadtxt(fpath)

        fcs = sliding_window_corrcoef(bold, window=window, padding=True)
        fcs = fcs + np.eye(fcs.shape[-1]) * 1e-4

        mms = ManifoldMeanShift(
            geometry='SPD', max_iter=max_iter, n_jobs=-1, n_clusters=n_clusters
        )
        mms.fit(fcs)
        labels, centers, shifted_points = (
            mms.labels,
            mms.cluster_centers,
            mms.shifted_points,
        )

        purity = purity_score(labels_true, labels)
        nmi = normalized_mutual_info_score(labels_true, labels)

        purities.update(purity)
        nmis.update(nmi)

        bar.set_description(f'purity: {purities.avg:.4f} | nmi: {nmis.avg:.4f}')
