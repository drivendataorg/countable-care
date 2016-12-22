#!/usr/bin/env python

from __future__ import division
from sklearn.cross_validation import StratifiedKFold
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import log_loss

import argparse
import logging
import numpy as np
import time

from kaggler.online_model import FM


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)


def logloss(y, p):
    p = max(1e-15, min(1 - 1e-15, p))
    return -np.log(p) if y == 1 else -np.log(1 - p)


def train_predict(train_file, test_file, predict_valid_file, predict_test_file,
                  n_iter=100, dim=4, lrate=.1, n_fold=5):

    _, y_val= load_svmlight_file(train_file)

    cv = StratifiedKFold(y_val, n_folds=n_fold, shuffle=True, random_state=2015)

    p_val = np.zeros_like(y_val)
    lloss = 0.
    for i_cv, (i_trn, i_val) in enumerate(cv, start=1):
        logging.info('CV training #{}'.format(i_cv))
        clf = FM(5e5, dim=dim, a=lrate)

        logging.info('Epoch\tTrain\tValid')
        logging.info('=========================')
        for i_iter in range(n_iter):
            lloss_trn = 0.
            cnt_trn = 0
            for i, (x, y) in enumerate(clf.read_sparse(train_file)):
                if i in i_val:
                    p_val[i] = clf.predict(x)
                else:
                    p = clf.predict(x)
                    clf.update(x, p - y)
                    lloss_trn += logloss(y, p)
                    cnt_trn += 1

            lloss_trn /= cnt_trn
            lloss_val = log_loss(y_val[i_val], p_val[i_val])

            if (i_iter == 0) or ((i_iter + 1) % int(n_iter / 10) == 0) or (i_iter == n_iter - 1):
                logging.info('#{:4d}\t{:.4f}\t{:.4f}'.format(i_iter + 1,
                                                             lloss_trn,
                                                             lloss_val))

        lloss += lloss_val

    logging.info('Log Loss = {:.4f}'.format(lloss / n_fold))

    logging.info('Retraining with 100% data...')
    clf = FM(5e5, dim=dim, a=lrate)
    for i_iter in range(n_iter):
        for x, y in clf.read_sparse(train_file):
            p = clf.predict(x)
            clf.update(x, p - y)

        logging.info('Epoch #{}'.format(i_iter + 1))

    _, y_tst = load_svmlight_file(test_file)
    p_tst = np.zeros_like(y_tst)
    for i, (x, _) in enumerate(clf.read_sparse(test_file)):
        p_tst[i] = clf.predict(x)

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
