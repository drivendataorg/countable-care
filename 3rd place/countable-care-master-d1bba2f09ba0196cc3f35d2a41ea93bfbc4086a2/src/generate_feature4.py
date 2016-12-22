#!/usr/bin/env python

from scipy import sparse
from sklearn.datasets import dump_svmlight_file

import argparse
import logging
import numpy as np
import os
import pandas as pd

from kaggler.util import encode_categorical_features


logging.basicConfig(format='%(asctime)s   %(levelname)s   %(message)s',
                    level=logging.DEBUG)


def generate_feature(train_file, label_file, test_file, feature_dir,
                     feature_name):
    # Load data files
    logging.info('Loading training and test data')
    trn = pd.read_csv(train_file, index_col=0)
    tst = pd.read_csv(test_file, index_col=0)
    label = pd.read_csv(label_file, index_col=0)
    n_trn = trn.shape[0]
    n_tst = tst.shape[0]

    logging.info('Combining training and test data')
    df = pd.concat([trn, tst], ignore_index=True)

    cols = list(df.columns)
    num_cols = [x for x in cols if x[0] == 'n']
    cat_cols = [x for x in cols if x[0] == 'c' or x[0] == 'r' or x[0] == 'o']

    # no transformation for numerical variables
    logging.info('Imputing missing values in numerical columns by 0')
    X_num = df.ix[:, num_cols].values
    X_num[np.isnan(X_num)] = 0.

    # One-Hot-Encoding for categorical variables
    logging.info('One-hot-encoding categorical columns')

    X_col = encode_categorical_features(df[cat_cols], min_obs=3, n=n_trn,
                                        nan_as_var=False)
    X = sparse.hstack((X_num, X_col))
    X = X.tocsr()

    logging.info('Saving features into {}'.format(feature_dir))
    for i in range(label.shape[1]):
        train_feature_file = os.path.join(feature_dir, '{}.trn{:02d}.sps'.format(feature_name, i))
        test_feature_file = os.path.join(feature_dir, '{}.tst{:02d}.sps'.format(feature_name, i))

        dump_svmlight_file(X[:n_trn], label.ix[:, i], train_feature_file,
                           zero_based=False)
        dump_svmlight_file(X[n_trn:], np.zeros((n_tst,)), test_feature_file,
                           zero_based=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', required=True, dest='train')
    parser.add_argument('--label-file', required=True, dest='label')
    parser.add_argument('--test-file', required=True, dest='test')
    parser.add_argument('--feature-dir', required=True, dest='feature_dir')
    parser.add_argument('--feature-name', required=True, dest='feature_name')

    args = parser.parse_args()

    generate_feature(train_file=args.train,
                     label_file=args.label,
                     test_file=args.test,
                     feature_dir=args.feature_dir,
                     feature_name=args.feature_name)
