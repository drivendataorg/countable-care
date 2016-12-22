#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier as RF

import argparse
import logging
import numpy as np
import time


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_est, depth, n_fold=5):

    logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                        level=logging.DEBUG, filename='rf_{}_{}.log'.format(
                                                        n_est, depth
                                                       ))

    logging.info('Loading training and test data...')
    X, y = load_svmlight_file(train_file)
    X_tst, _ = load_svmlight_file(test_file)

    clf = RF(n_estimators=n_est, max_depth=depth, random_state=2015)

    cv = StratifiedKFold(y, n_folds=n_fold, shuffle=True, random_state=2015)

    logging.info('Cross validation...')
    p_val = np.zeros_like(y)
    lloss = 0.
    for i_trn, i_val in cv:
        clf.fit(X[i_trn], y[i_trn])
        p_val[i_val] = clf.predict_proba(X[i_val])[:, 1]
        lloss += log_loss(y[i_val], p_val[i_val])

    logging.info('Log Loss = {:.4f}'.format(lloss))

    logging.info('Retraining with 100% data...')
    clf.fit(X.todense(), y)
    p_tst = clf.predict_proba(X_tst.todense())[:, 1]

    logging.info('Saving predictions...')
    np.savetxt(predict_valid_file, p_val, fmt='%.6f')
    np.savetxt(predict_test_file, p_tst, fmt='%.6f')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train_file')
    parser.add_argument('--test-file', required=True, dest='test_file')
    parser.add_argument('--predict-valid-file', required=True,
                        dest='predict_valid_file')
    parser.add_argument('--predict-test-file', required=True,
                        dest='predict_test_file')
    parser.add_argument('--n-est', default=100, type=int, dest='n_est')
    parser.add_argument('--depth', default=None, type=int, dest='depth')

    args = parser.parse_args()

    start = time.time()
    train_predict(train_file=args.train_file,
                  test_file=args.test_file,
                  predict_valid_file=args.predict_valid_file,
                  predict_test_file=args.predict_test_file,
                  n_est=args.n_est,
                  depth=args.depth)
    logging.info('finished ({:.2f} min elasped)'.format((time.time() - start) /
                                                        60))
