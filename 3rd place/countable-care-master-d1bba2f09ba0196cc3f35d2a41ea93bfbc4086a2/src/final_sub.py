#!/usr/bin/env python
import argparse
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--first-model', required=True, dest='first')
    parser.add_argument('--second-model', required=True, dest='second')
    parser.add_argument('--out-file', required=True, dest='out')

    args = parser.parse_args()

    y1 = pd.read_csv(args.first, index_col=0)
    y2 = pd.read_csv(args.second, index_col=0)

    y = .7 * y1 + .3 * y2
    y.to_csv(args.out)
