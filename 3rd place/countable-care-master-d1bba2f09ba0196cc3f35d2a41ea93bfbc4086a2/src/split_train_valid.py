#!/usr/bin/env python

from itertools import izip
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--feature-file', required=True, dest='feature')
    parser.add_argument('--valid-train-file', required=True, dest='valid_train')
    parser.add_argument('--valid-test-file', required=True, dest='valid_test')
    parser.add_argument('--train-id', required=True, dest='train_id')
    parser.add_argument('--valid-test-id', required=True, dest='valid_test_id')

    args = parser.parse_args()

    with open(args.valid_test_id) as f:
        valid_ids = set([int(x.strip()) for x in f.readlines()])

    with open(args.train_id) as f_train_id, open(args.feature) as f_feature, \
         open(args.valid_train, 'w') as f_trn, \
         open(args.valid_test, 'w') as f_val:

        for train_id, row in izip(f_train_id, f_feature):
            train_id = int(train_id.strip())
            if train_id in valid_ids:
                f_val.write(row)
            else:
                f_trn.write(row)

