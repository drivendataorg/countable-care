# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 11:08:31 2013

"""
#################################################
####http://wiki.scipy.org/NumPy_for_Matlab_Users
#################################################


# -*- coding: utf-8 -*-
import sys
import numpy as np
from numpy import array, hstack
from sklearn import metrics,preprocessing,cross_validation

from sklearn.linear_model import LogisticRegression
import pandas as p

from sklearn.cross_validation import KFold

import codecs
import pickle
import gc
from math import log
from sklearn.feature_extraction.text import TfidfVectorizer

def ll( target , predicted ):
     if len(predicted) != len(target):
         print 'lengths not equal!'
         return
     predicted = [min([max([x,1e-15]),1-1e-15]) for x in predicted]  # within (0,1) interval
     return -(1.0/len(target))*sum([target[i]*log(predicted[i]) + (1.0-target[i])*log(1.0-predicted[i]) for i in xrange(len(target))])

def LOGLOSS( target , predicted  ):
    s=0.
    for i in range(0,14):
        print i, ll( target[:,i] , predicted[:,i]  )
        s =  s + ll( target[:,i] , predicted[:,i]  )
    return( s/14 )


##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
# parameters #################################################################
train = 'input//train.csv'  #path to training file
label = 'input//target.csv'
test  = 'input//test.csv'    # path to testing file

print 'Starting...'

print 'Loading train...'
tr  = []
i = -1
for line in open(train):

    row = line.rstrip().split(',')

    s = ''
    for n in range(len(row)):
        s = s + 'f' + str(n) + '_' + row[n][:4] + ' '
    tr.append( s )
#    while 1: 1
    i += 1
print len(tr)
print tr[0]

print 'Loading test...'
ts  = []
i = -1
for line in open(test):
    row = line.rstrip().split(',')
    s = ''
    for n in range(len(row)):
        s = s + 'f' + str(n) + '_' + row[n][:4] + ' '
    ts.append( s )
    i += 1
print len(ts)
print ts[0]

print 'TfidfVectorizer...'
tfidf = TfidfVectorizer(stop_words=None, sublinear_tf=True, smooth_idf=True,
                        use_idf=True, min_df=2, max_df=0.99, strip_accents='unicode',
                        analyzer='word', ngram_range=(1,2))

Xall = tfidf.fit_transform( tr+ts )
print Xall[0,:]
print Xall.shape

ts = Xall[len(tr):,:]
tr = Xall[:len(tr),:]
del Xall
gc.collect()

with open('data2.pik', 'wb') as f:
    pickle.dump([tr,ts], f, -1)
f.close()
gc.collect()

print 'Done!'
##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################





##########################################################################################
##########################################################################################
##########################################################################################
##########################################################################################
with open('data2.pik', 'rb') as f:
    X,Xtest = pickle.load(f)


print X.shape
print Xtest.shape
gc.collect()

NFOLDs = 4
cv  = [ i % NFOLDs for i in range(X.shape[0]) ]

predtrain = np.ndarray( (X.shape[0],14) )
predtest  = np.ndarray( (Xtest.shape[0],14) )
predtrain = np.ndarray( X.shape[0] )
Y = np.ndarray( X.shape[0] )
Y = np.loadtxt('input/target.csv', delimiter=",")

PRED = np.zeros( (X.shape[0],14) )

cost = [4.5,1.5,1.4,21,4.5,12,4.5,2.5,10,1.25,1.1,21,25,1.5 ]
for TARGET in range(14):
    print "Target:", TARGET
    rd = LogisticRegression(C= cost[TARGET]  , dual=True, class_weight=None )
    print rd.get_params()

    for fold in range(NFOLDs):
        px = [ n  for n in range(X.shape[0]) if cv[n]!=fold ]
        py = [ n  for n in range(X.shape[0]) if cv[n]==fold ]
        trainx, valx = X[px], X[py]
        trainy, valy = Y[px,TARGET], Y[py,TARGET]

        rd.fit( trainx, trainy )
        pred = rd.predict_proba( valx )[:,1]
        predtrain[py] = pred
        print fold, ll( valy , pred )

    print str(TARGET),' => ',ll( Y[:,TARGET] , predtrain )
    PRED[:,TARGET] = predtrain
    rd.fit( X, Y[:,TARGET] )
    predtest = rd.predict_proba(Xtest)[:,1]
    np.savetxt( 'predictions/predtrain2_'+str(TARGET)+'.csv',predtrain,delimiter=',' )
    np.savetxt( 'predictions/predtest2_'+ str(TARGET)+'.csv',predtest ,delimiter=',' )

print "ALL:\n", LOGLOSS(  Y , PRED   )

print 'Done!!!'

#ALL:
#0 0.362388574671
#1 0.49744510127
#2 0.503690493404
#3 0.0339081809014
#4 0.161497773342
#5 0.0688543312854
#6 0.169007969301
#7 0.457791044066
#8 0.0642436749377
#9 0.36682939946
#10 0.461284183194
#11 0.100247960818
#12 0.0943959990438
#13 0.3431110549
#0.2631925529
