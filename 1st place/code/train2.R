rm(list = ls(all = TRUE))
source("fn.base.R")
library(xgboost)
library(data.table)
library(foreach)
gc()

fn.load.data("train")
fn.load.data("test")
target <- fread( 'input/train_labels.csv' )
str(train)


cs <- colSums(train == -999)
cs <- order(cs)
train <- train[ , cs]
test  <- test[ , cs]
gc()

TGT <- as.matrix( target )
TGT <- TGT[,2:ncol(TGT)]

train <- train[,1:1365]
test <- test[,1:1365]

cv <- rep( 1:4 , length.out=nrow(train)   )

FEAT = 1
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1000 ,shri=0.006, depth=10,subsample=0.750,colsample=0.50, verbose=TRUE, trainFULL=TRUE )
feats <- 10:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1000 ,shri=0.006, depth=10,subsample=0.750,colsample=0.50, verbose=TRUE, trainFULL=TRUE )
feats <- 20:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1000 ,shri=0.006, depth=10,subsample=0.750,colsample=0.50, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_1 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_1")

FEAT = 2
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1000 ,shri=0.009, depth=5,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
feats <- 10:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1000 ,shri=0.009, depth=5,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
feats <- 5:1365
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1000 ,shri=0.009, depth=7,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_2 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_2")

FEAT = 3
feats <- 1:1345
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1000 ,shri=0.009, depth=7,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
feats <- 10:1355
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1000 ,shri=0.009, depth=7,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
feats <- 20:1365
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1000 ,shri=0.009, depth=7,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_3 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_3")

FEAT = 4
feats <- 15:1350
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=400 ,shri=0.022, depth=4,subsample=0.750,colsample=0.2750, verbose=TRUE, trainFULL=TRUE )
feats <- 10:1345
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=400 ,shri=0.02, depth=4,subsample=0.850,colsample=0.300, verbose=TRUE, trainFULL=TRUE )
feats <- 15:1355
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.80,colsample=0.250, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_4 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_4")

FEAT = 5
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=5,subsample=0.750,colsample=0.2750, verbose=TRUE, trainFULL=TRUE )
feats <- 5:660
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=6,subsample=0.750,colsample=0.2750, verbose=TRUE, trainFULL=TRUE )
feats <- 10:770
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=7,subsample=0.750,colsample=0.2750, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_5 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_5")

FEAT = 6
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 5:660
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=5,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 10:770
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=5,subsample=0.7750,colsample=0.33, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_6 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_6")

FEAT = 7
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 5:660
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=5,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 10:770
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=5,subsample=0.7750,colsample=0.33, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_7 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_7")

FEAT = 8
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 5:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.7750,colsample=0.33, verbose=TRUE, trainFULL=TRUE )
feats <- 10:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.80,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_8 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_8")

FEAT = 9
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 5:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.7750,colsample=0.33, verbose=TRUE, trainFULL=TRUE )
feats <- 10:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.80,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_9 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_9")

FEAT = 10
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 10:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 20:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=5,subsample=0.80,colsample=0.350, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_10 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_10")

FEAT = 11
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 10:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 20:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.01, depth=5,subsample=0.80,colsample=0.350, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_11 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_11")

FEAT = 12
feats <- 1:650
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.015, depth=8,subsample=0.750,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
feats <- 5:1000
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.015, depth=8,subsample=0.750,colsample=0.45, verbose=TRUE, trainFULL=TRUE )
feats <- 10:750
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.015, depth=8,subsample=0.80,colsample=0.40, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_12 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_12")

FEAT = 13
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.015, depth=8,subsample=0.750,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
feats <- 5:650
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.015, depth=8,subsample=0.750,colsample=0.45, verbose=TRUE, trainFULL=TRUE )
feats <- 10:750
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.015, depth=8,subsample=0.80,colsample=0.40, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_13 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_13")

FEAT = 14
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.010, depth=8,subsample=0.750,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
feats <- 5:650
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.009, depth=8,subsample=0.750,colsample=0.45, verbose=TRUE, trainFULL=TRUE )
feats <- 10:750
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=800 ,shri=0.0085, depth=8,subsample=0.80,colsample=0.40, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg9_14 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg9_14")


pred <- TGT
pred[ , 1 ] <- xg9_1$train
pred[ , 2 ] <- xg9_2$train
pred[ , 3 ] <- xg9_3$train
pred[ , 4 ] <- xg9_4$train
pred[ , 5 ] <- xg9_5$train
pred[ , 6 ] <- xg9_6$train
pred[ , 7 ] <- xg9_7$train
pred[ , 8 ] <- xg9_8$train
pred[ , 9 ] <- xg9_9$train
pred[ , 10 ] <- xg9_10$train
pred[ , 11 ] <- xg9_11$train
pred[ , 12 ] <- xg9_12$train
pred[ , 13 ] <- xg9_13$train
pred[ , 14 ] <- xg9_14$train
llfun( TGT , pred  )
llfun( TGT[,1] , pred[,1]  )
llfun( TGT[,2] , pred[,2]  )
llfun( TGT[,3] , pred[,3]  )
llfun( TGT[,4] , pred[,4]  )
llfun( TGT[,5] , pred[,5]  )
llfun( TGT[,6] , pred[,6]  )
llfun( TGT[,7] , pred[,7]  )
llfun( TGT[,8] , pred[,8]  )
llfun( TGT[,9] , pred[,9]  )
llfun( TGT[,10] , pred[,10]  )
llfun( TGT[,11] , pred[,11]  )
llfun( TGT[,12] , pred[,12]  )
llfun( TGT[,13] , pred[,13]  )
llfun( TGT[,14] , pred[,14]  )
#0.2511395
# [1] 0.3319077
# [1] 0.4828697
# [1] 0.4938821
# [1] 0.02757748
# [1] 0.1540029
# [1] 0.05920316
# [1] 0.1668277
# [1] 0.4351065
# [1] 0.05173878
# [1] 0.3609212
# [1] 0.4530481
# [1] 0.0854969
# [1] 0.07725514
# [1] 0.3361163


sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub$service_a <- xg9_1$test
sub$service_b <- xg9_2$test
sub$service_c <- xg9_3$test
sub$service_d <- xg9_4$test
sub$service_e <- xg9_5$test
sub$service_f <- xg9_6$test
sub$service_g <- xg9_7$test
sub$service_h <- xg9_8$test
sub$service_i <- xg9_9$test
sub$service_j <- xg9_10$test
sub$service_k <- xg9_11$test
sub$service_l <- xg9_12$test
sub$service_m <- xg9_13$test
sub$service_n <- xg9_14$test

pred.xg.9 <- list(  train=pred , test=as.matrix(sub[ , 2:ncol(sub)  ]) )
fn.save.data("pred.xg.9")
str( pred.xg.9$train  )
str( pred.xg.9$test  )

# write.table( sub , "submission/sub9.csv", row.names=FALSE, quote=FALSE,sep="," )

