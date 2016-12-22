#!/bin/bash
echo 'Training FM(200, 4, 0.001) on feature2'
make -f Makefile.fm_200_4_0.001_feature2
echo 'Training FM(200, 8, 0.001) on feature3'
make -f Makefile.fm_200_8_0.001_feature3
echo 'Training GBM+Bagging(40, 7, 0.1) on feature10'
make -f Makefile.gbm_bagging_40_7_0.1_feature10
echo 'Training libFM(200, 4, 0.005) on feature2'
make -f Makefile.libfm_200_4_0.005_feature2
echo 'Training libFM(200, 4, 0.005) on feature4'
make -f Makefile.libfm_200_4_0.005_feature4
echo 'Training LR(0.1) on feature2'
make -f Makefile.lr_0.1_feature2
echo 'Training LR(0.1) on feature4'
make -f Makefile.lr_0.1_feature4
echo 'Training NN(20, 64, 0.005) on feature8'
make -f Makefile.nn_20_64_0.005_feature8
echo 'Training NN(20, 8, 0.01) on feature2'
make -f Makefile.nn_20_8_0.01_feature2
echo 'Training NN(20, 8, 0.01) on feature3'
make -f Makefile.nn_20_8_0.01_feature3
echo 'Training RF(400, 40) on feature10'
make -f Makefile.rf_400_40_feature10
echo 'Training RF(400, 40) on feature2'
make -f Makefile.rf_400_40_feature2
echo 'Training RF(400, 40) on feature5'
make -f Makefile.rf_400_40_feature5
echo 'Training RF(400, 40) on feature9'
make -f Makefile.rf_400_40_feature9
echo 'Training XG(100, 8, 0.05) on feature1'
make -f Makefile.xg_100_8_0.05_feature1
echo 'Training XG(100, 8, 0.05) on feature10'
make -f Makefile.xg_100_8_0.05_feature10
echo 'Training XG(100, 8, 0.05) on feature8'
make -f Makefile.xg_100_8_0.05_feature8
echo 'Training XG(100, 8, 0.05) on feature9'
make -f Makefile.xg_100_8_0.05_feature9
echo 'Training XG+Bagging(120, 7, 0.1) on feature9'
make -f Makefile.xg_bagging_120_7_0.1_feature9

echo 'Training Ensemble Model - XG(Grid Searched Param) on esb19'
make -f Makefile.esb.xg_grid_colsub

echo 'Combining the Ensemble Model with Prediction of Abhishek'
python src/final_sub.py --first-model build/tst/esb_xg_grid_colsub.sub.csv \
                        --second-model abhishek/sub_xgb120.csv \
                        --out-file build/tst/final_sub.csv
