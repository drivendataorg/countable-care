#!/usr/bin/env python

from __future__ import division
from datetime import datetime
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import dump_svmlight_file, load_svmlight_file
from sklearn.metrics import log_loss

import argparse
import logging
import numpy as np
import os
import subprocess
import time


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_iter=100, dim=4, lrate=.1, n_fold=5):

    feature_name = os.path.basename(train_file)[:-8]
    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename='libfm_{}_{}_{}_{}.log'.format(
                                                        n_iter, dim, lrate,
                                                        feature_name
                                                      ))

    logging.info('Loading training data')
    X, y = load_svmlight_file(train_file)

    cv = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=2015)

    logging.info('Cross validation...')
    p = np.zeros_like(y)
    lloss = 0.
    for i_trn, i_val in cv:
        now = datetime.now().strftime('%Y%m%d-%H%M%S')
        valid_train_file = '/tmp/libfm_train_{}_{}.sps'.format(feature_name, now)
        valid_test_file = '/tmp/libfm_valid_{}_{}.sps'.format(feature_name, now)
        valid_predict_file = '/tmp/libfm_predict_{}_{}.sps'.format(feature_name, now)

        dump_svmlight_file(X[i_trn], y[i_trn], valid_train_file,
                           zero_based=False)
        dump_svmlight_file(X[i_val], y[i_val], valid_test_file,
                           zero_based=False)

        subprocess.call(["libFM",
                         "-task", "c",
                         '-dim', '1,1,{}'.format(dim),
                         '-init_stdev', str(lrate),
                         '-iter', str(n_iter),
                         '-train', valid_train_file,
                         '-test', valid_test_file,
                         '-out', valid_predict_file])

        p[i_val] = np.loadtxt(valid_predict_file)
        lloss += log_loss(y[i_val], p[i_val])

        os.remove(valid_train_file)
        os.remove(valid_test_file)
        os.remove(valid_predict_file)

    logging.info('Log Loss = {:.4f}'.format(lloss / n_fold))
    np.savetxt(predict_valid_file, p, fmt='%.6f')

    logging.info('Retraining with 100% data...')
    subprocess.call(["libFM",
                     "-task", "c",
                     '-dim', '1,1,{}'.format(dim),
                     '-init_stdev', str(lrate),
                     '-iter', str(n_iter),
                     '-train', train_file,
                     '-test', test_file,
                     '-out', predict_test_file])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-iter', type=int, dest='n_iter')
    parser.add_argument('--dim', type=int, dest='dim')
    parser.add_argument('--lrate', type=float, dest='lrate')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_iter=args.n_iter,
                  dim=args.dim,
                  lrate=args.lrate)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
