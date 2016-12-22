# Countable Care: Modeling Women's Care Decisions - 1st Place

## Entrant Background and Submission Overview

### Mini-bio
I have a MS in Electronic Engineering. In the past 16 years I've been working as an Electronic Engineer for big multinationals. In 2008 I started teaching myself the techniques of machine learning, by first turning to improving personal stock market gains and later learning a lot of techniques to improve generic predictions. In 2012, I joined the Kaggle competition site and won a couple of competitions and achieved good results in some others, improving my machine learning skills in many areas.

### High Level Summary of Submission
My approach is based in an ensemble of models. To make it possible I trained a lot of models, with different hyper-parameters, number of features and different learning techniques. That was the first level of learning. All training is made using a 4 fold cross validation technique, where it generates two predictions sets: one cross validated train set and one test set for each model. All those models are then ensembled using a second level of learning, training with all that cross validated train sets (meta features), and the final model is applied in the predicted test sets.  That approach was chosen because the dataset is composed of 1379 features, most of it categorical and training instances are not big enough to do a reliable feature selection. As the predictive power of the dataset comes mainly from the categorical features and features don’t have many levels, I decided to quit using a simple feature selection and use the ensemble technique to better explore feature interaction between all features and different learning techniques. My final ensemble was composed by 9 models:
* 1 Random Forest
* 6 Xgboost
* 2 Logistic Regression

### Omitted Work
I tried to make models for numeric, ordinal and categorical variables separately, then ensemble all predictions, but performance decreased. It seems there are some levels of iteration between these classes of features.
I also tried some other training algorithms like Factorization Machines, Vowpal Wabbit and FTRL, but performance was not good enough to use it.

### Tools Used
Data preparation, analysis and training was done in R and Python scikit was used to train Logistic Regression models.

### Model Evaluation
I evaluated the performance using a 4 fold cross validation in trainset using the provided metric. The models train4.R to train7.R used a preprocessing, treating all features as categorical and applying a crossvalidated linear regression to achieve the individual performance of each feature. I then choose the features used for training algorithms based in that performance.

### Future Steps
* Try tuning the models using a grid search of hyper-parameters and for selecting features. It’s going to be very slow, but it can help a lot in the final ensemble.
* Explore ensembling different training algorithms.


## Replicating the Submission
Instruction for Linux Ubuntu 14.04, probably also runs under Windows.
Code run ok in a i7 8 core CPU with 24GB RAM.

### Install R Language
* Install R Studio
* Open R studio
* Install packages:
	* xgboost
	* data.table
	* randomForest
	* doSNOW
	* foreach
	* gbm
	* Matrix and all dependencies packages asks for.

### Install Python
* Install scikit

### Format Folder Structure
* Place Datasets in /input folder
* Folder /debug will be used for debuging Random Forest parallel processing
* Folder /data will be used to store variables
* Folder /predictions will be used to store python models
* Folder /submission will be used to place the final model predictions.

### Run R script train1.R    
   This script will load datasets, preprocess trainset and testset setting -999 to not available(NA) features, compute randomForest importance for each of the 14 targets, train using randomForest all 14 targets over 4 folds, each target trained 3 models with different number of features based in the randomForest importances, bag 3 models for each target and then saves the results (pred.rf.10).

### Run R script train2.R    
   This script will load datasets, preprocess trainset and testset, train using Xgboost all 14 targets over 4 folds, bag 3 models for each target and then saves the results (pred.xg.9).

### Run R script train3.R    
   This script will load datasets, remove features levels that appear only once, train using Xgboost all 14 targets over 4 folds, bag 3 models for each target and then saves the results (pred.xg.12).

### Run R script train4.R    
   This script will load datasets, preprocess trainset and testset, turn into probabilities using glm on each feature independently for each one of the 14 targets, train using Xgboost all 14 targets over 4 folds using that probabilities converted trainset, then saves the results (pred.xg.13).

### Run R script train5.R
  Same as train4.R, but using different hyperparameters and features. Saves the results (pred.xg.17)

### Run R script train6.R
  Same as train4.R, but using different hyperparameters and features. Saves the results (pred.xg.19)

### Run R script train7.R
  Same as train4.R, but using different hyperparameters and features. Saves the results (pred.xg.20)

### Run Python script train1.py    
   This script will load datasets, turn all features into categorical and sparse, train all 14 targets over 4 folds using Logistic Regression, then saves the results.

### Run Python script train2.py    
   This script will load datasets, turn all features into categorical and sparse, convert to 1 and 2 way NGRAMs, train all 14 targets over 4 folds using Logistic Regression, then saves the results.

### Run script ensembleALL.R
   This script will load previous model predictions to be used as meta features in a second level of training. Then ensemble all using Xgboost, run three ensemble models using different input models. Then apply a arithmetic mean over all 3 models, write submission file (submission/final_aritmetic_ensemble.csv). Local CV printed should be around 0.2464.

Some hours later...DONE!!!
