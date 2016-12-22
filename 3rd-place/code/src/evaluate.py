#!/usr/bin/env python

from __future__ import division
from sklearn.metrics import log_loss, roc_auc_score

import argparse
import numpy as np
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-file', '-t', required=True, dest='target_file')
    parser.add_argument('--predict-file', '-p', required=True, dest='predict_file')
    args = parser.parse_args()

    p = np.loadtxt(args.predict_file, delimiter=',')
    y = np.loadtxt(args.target_file, delimiter=',')

    model_name = os.path.basename(args.predict_file)[:-8]

    n_class = p.shape[1]
    lloss = 0.
    auc = 0.
    for i in range(n_class):
        lloss += log_loss(y[:, i], p[:, i])
        auc += roc_auc_score(y[:, i], p[:, i])

    print('{}\t{:.4f}\t{:.4f}'.format(model_name, lloss / n_class, auc / n_class))
