rm(list = ls(all = TRUE))
source("fn.base.R")
library(xgboost)
library(data.table)
library(randomForest)
library(foreach)
library(Matrix)
gc()

fn.load.data("train")
fn.load.data("test")
target <- fread( 'input/train_labels.csv' )
TGT <- as.matrix( target )
TGT <- TGT[,2:ncol(TGT)]

train <- as.data.frame(train)
test  <- as.data.frame(test)
cv <- rep(1:4,length.out=nrow(train))
px <- 1:nrow(train)
py <- (nrow(train)+1):(nrow(train)+nrow(test))
set.seed(11111)

l <- which( colSd(train)!=0 )
train <- train[,l]
test  <- test[,l]


train <- as.matrix(train)
test  <- as.matrix(test)

############################################################################
############################################################################
############################################################################
target <- fread( 'input/train_labels.csv' )
TGT <- as.matrix( target )
TGT <- TGT[,2:ncol(TGT)]

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ]

tgt =1
cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:ncol(train)
xg20_1 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv,
                  ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_1$train  )
fn.save.data("xg20_1")
#
############################################################################
############################################################################
############################################################################
tgt = 2
feats <- 1:ncol(train)

xg20_2 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_2$train  )
fn.save.data("xg20_2")
#
############################################################################
############################################################################
############################################################################
tgt= 3
feats <- 1:ncol(train)

xg20_3 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_3$train  )
fn.save.data("xg20_3")
#
############################################################################
############################################################################
############################################################################
tgt =8
feats <- 1:ncol(train)

xg20_8 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_8$train  )
fn.save.data("xg20_8")
#
############################################################################
############################################################################
############################################################################
tgt =11
feats <- 1:ncol(train)

xg20_11 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_11$train  )
fn.save.data("xg20_11")
#
############################################################################
############################################################################
############################################################################
tgt = 4
feats <- 1:ncol(train)

xg20_4 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_4$train  )
fn.save.data("xg20_4")
#
############################################################################
############################################################################
############################################################################
tgt = 5
feats <- 1:ncol(train)

xg20_5 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_5$train  )
fn.save.data("xg20_5")
#
############################################################################
############################################################################
############################################################################
tgt = 6
feats <- 1:ncol(train)

xg20_6 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_6$train  )
fn.save.data("xg20_6")
#
############################################################################
############################################################################
############################################################################
tgt = 7
feats <- 1:ncol(train)

xg20_7 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_7$train  )
fn.save.data("xg20_7")
#
############################################################################
############################################################################
############################################################################
tgt = 9
feats <- 1:ncol(train)

xg20_9 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_9$train  )
fn.save.data("xg20_9")
#
############################################################################
############################################################################
############################################################################
tgt = 10
feats <- 1:ncol(train)

xg20_10 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_10$train  )
fn.save.data("xg20_10")
#
############################################################################
############################################################################
############################################################################
tgt = 12
feats <- 1:ncol(train)

xg20_12 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_12$train  )
fn.save.data("xg20_12")
#
############################################################################
############################################################################
############################################################################
tgt = 13
feats <- 1:ncol(train)

xg20_13 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_13$train  )
fn.save.data("xg20_13")
#
############################################################################
############################################################################
############################################################################
tgt = 14
feats <- 1:ncol(train)

xg20_14 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1500 ,shri=0.015, depth=6,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg20_14$train  )
fn.save.data("xg20_14")


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
fn.load.data("xg20_1")
fn.load.data("xg20_2")
fn.load.data("xg20_3")
fn.load.data("xg20_4")
fn.load.data("xg20_5")
fn.load.data("xg20_6")
fn.load.data("xg20_7")
fn.load.data("xg20_8")
fn.load.data("xg20_9")
fn.load.data("xg20_10")
fn.load.data("xg20_11")
fn.load.data("xg20_12")
fn.load.data("xg20_13")
fn.load.data("xg20_14")
###############################################################################################
pred <- TGT
pred[ , 1 ] <- xg20_1$train
pred[ , 2 ] <- xg20_2$train
pred[ , 3 ] <- xg20_3$train
pred[ , 4 ] <- xg20_4$train
pred[ , 5 ] <- xg20_5$train 
pred[ , 6 ] <- xg20_6$train
pred[ , 7 ] <- xg20_7$train
pred[ , 8 ] <- xg20_8$train
pred[ , 9 ] <- xg20_9$train
pred[ , 10 ] <- xg20_10$train
pred[ , 11 ] <- xg20_11$train
pred[ , 12 ] <- xg20_12$train
pred[ , 13 ] <- xg20_13$train
pred[ , 14 ] <- xg20_14$train
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
#0.2543649
# [1] 0.3346353
# [1] 0.4856011
# [1] 0.4959782
# [1] 0.03001092
# [1] 0.1571497
# [1] 0.06544206
# [1] 0.1747869
# [1] 0.4352492
# [1] 0.05891442
# [1] 0.3650521
# [1] 0.4575974
# [1] 0.0855437
# [1] 0.07741901
# [1] 0.3377286


sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub$service_a <- xg20_1$test
sub$service_b <- xg20_2$test
sub$service_c <- xg20_3$test
sub$service_d <- xg20_4$test
sub$service_e <- xg20_5$test
sub$service_f <- xg20_6$test
sub$service_g <- xg20_7$test
sub$service_h <- xg20_8$test
sub$service_i <- xg20_9$test
sub$service_j <- xg20_10$test
sub$service_k <- xg20_11$test
sub$service_l <- xg20_12$test
sub$service_m <- xg20_13$test
sub$service_n <- xg20_14$test

pred.xg.20 <- list(  train=pred , test=as.matrix(sub[ , 2:ncol(sub)  ]) )
fn.save.data("pred.xg.20")
str( pred.xg.20$train  )
str( pred.xg.20$test  )

# write.table( sub , "submission/sub20.csv", row.names=FALSE, quote=FALSE,sep="," )

