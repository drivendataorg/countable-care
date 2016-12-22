#!/usr/bin/env python

from scipy import sparse
from sklearn.datasets import dump_svmlight_file

import argparse
import logging
import numpy as np
import os
import pandas as pd


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)


def ensemble_feature(valid_file, label_file, test_file, feature_dir,
                     feature_name):
    # Load data files
    logging.info('Loading training and test data')
    val = np.loadtxt(valid_file, delimiter=',')
    tst = np.loadtxt(test_file, delimiter=',')
    label = np.loadtxt(label_file, delimiter=',')

    logging.info('{}x{}, {}x{}'.format(val.shape[0], val.shape[1],
                                       tst.shape[0], tst.shape[1]))

    n_val = val.shape[0]
    n_tst = tst.shape[0]

    logging.info('Saving features into {}'.format(feature_dir))
    for i in range(label.shape[1]):
        train_feature_file = os.path.join(feature_dir, '{}.trn{:02d}.sps'.format(feature_name, i))
        test_feature_file = os.path.join(feature_dir, '{}.tst{:02d}.sps'.format(feature_name, i))

        dump_svmlight_file(val, label[:, i], train_feature_file,
                           zero_based=False)
        dump_svmlight_file(tst, np.zeros((n_tst,)), test_feature_file,
                           zero_based=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--valid-file', required=True, dest='valid')
    parser.add_argument('--label-file', required=True, dest='label')
    parser.add_argument('--test-file', required=True, dest='test')
    parser.add_argument('--feature-dir', required=True, dest='feature_dir')
    parser.add_argument('--feature-name', required=True, dest='feature_name')

    args = parser.parse_args()

    ensemble_feature(valid_file=args.valid,
                     label_file=args.label,
                     test_file=args.test,
                     feature_dir=args.feature_dir,
                     feature_name=args.feature_name)
