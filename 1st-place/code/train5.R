rm(list = ls(all = TRUE))
source("fn.base.R")
library(xgboost)
library(data.table)
library(randomForest)
library(e1071)
library(foreach)
library(gbm)
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


############################################################################
############################################################################
############################################################################
target <- fread( 'input/train_labels.csv' )
TGT <- as.matrix( target )
TGT <- TGT[,2:ncol(TGT)]

fn.load.data("tr1");
fn.load.data("ts1");
train <- tr1
test  <- ts1
rm(tr1,ts1);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ]

tgt =1
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_1 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv,
                  ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_1$train  )
fn.save.data("xg17_1")
#
#
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr2");
fn.load.data("ts2");
train <- tr2
test  <- ts2
rm(tr2,ts2);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 2
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- c( 1 , which( sc < 1.00*median(sc[2:ncol(train)]) ) )
feats <- 1:1000
xg17_2 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_2$train  )
fn.save.data("xg17_2")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr3");
fn.load.data("ts3");
train <- tr3
test  <- ts3
rm(tr3,ts3);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt= 3
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- c( 1 , which( sc < 1.00*median(sc[2:ncol(train)]) ) )
feats <- 1:1000
xg17_3 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_3$train  )
fn.save.data("xg17_3")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr8");
fn.load.data("ts8");
train <- tr8
test  <- ts8
rm(tr8,ts8);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt =8
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_8 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_8$train  )
fn.save.data("xg17_8")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr11");
fn.load.data("ts11");
train <- tr11
test  <- ts11
rm(tr11,ts11);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt =11
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_11 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_11$train  )
fn.save.data("xg17_11")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr4");
fn.load.data("ts4");
train <- tr4
test  <- ts4
rm(tr4,ts4);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 4
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_4 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_4$train  )
fn.save.data("xg17_4")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr5");
fn.load.data("ts5");
train <- tr5
test  <- ts5
rm(tr5,ts5);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 5
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
plot(sc[1:ncol(train)])
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_5 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_5$train  )
fn.save.data("xg17_5")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr6");
fn.load.data("ts6");
train <- tr6
test  <- ts6
rm(tr6,ts6);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 6
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_6 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_6$train  )
fn.save.data("xg17_6")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr7");
fn.load.data("ts7");
train <- tr7
test  <- ts7
rm(tr7,ts7);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 7
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_7 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_7$train  )
fn.save.data("xg17_7")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr9");
fn.load.data("ts9");
train <- tr9
test  <- ts9
rm(tr9,ts9);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 9
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_9 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_9$train  )
fn.save.data("xg17_9")
#
#
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr10");
fn.load.data("ts10");
train <- tr10
test  <- ts10
rm(tr10,ts10);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 10
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_10 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
#                   ite=5000 ,shri=0.007135, depth=3,subsample=0.80,colsample=0.30, verbose=FALSE, trainFULL=TRUE, ncores=7 )
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_10$train  )
fn.save.data("xg17_10")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr12");
fn.load.data("ts12");
train <- tr12
test  <- ts12
rm(tr12,ts12);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 12
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_12 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
#                    ite=5000 ,shri=0.00510, depth=3,subsample=0.800,colsample=0.300, verbose=TRUE, trainFULL=TRUE, ncores=7 )
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_12$train  )
fn.save.data("xg17_12")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr13");
fn.load.data("ts13");
train <- tr13
test  <- ts13
rm(tr13,ts13);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 13
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_13 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
#                    ite=5000 ,shri=0.007135, depth=3,subsample=0.800,colsample=0.300, verbose=TRUE, trainFULL=TRUE, ncores=7 )
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_13$train  )
fn.save.data("xg17_13")
#
############################################################################
############################################################################
############################################################################
fn.load.data("tr14");
fn.load.data("ts14");
train <- tr14
test  <- ts14
rm(tr14,ts14);gc()

l <- as.numeric( lapply( lapply(as.data.frame(train),unique) , length  ) )
train <- train[ , which(l>1) ]
test  <- test[ , which(l>1) ];gc()

tgt = 14
sc <- rep( 0 , ncol(train) )
for( i in 1:ncol(train)){
  sc[i] <- llfun( TGT[,tgt] , train[,i] )
}
o <- order( sc , decreasing=FALSE )
train <- train[,o]
test  <- test[,o]
sc <- sc[o]
plot(sc[1:ncol(train)])

cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- 1:1000
xg17_14 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
#                    ite=5000 ,shri=0.007135, depth=3,subsample=0.80,colsample=0.300, verbose=TRUE, trainFULL=TRUE, ncores=7 )
                    ite=1000 ,shri=0.02, depth=5,subsample=0.90,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg17_14$train  )
fn.save.data("xg17_14")


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
fn.load.data("xg17_1")
fn.load.data("xg17_2")
fn.load.data("xg17_3")
fn.load.data("xg17_4")
fn.load.data("xg17_5")
fn.load.data("xg17_6")
fn.load.data("xg17_7")
fn.load.data("xg17_8")
fn.load.data("xg17_9")
fn.load.data("xg17_10")
fn.load.data("xg17_11")
fn.load.data("xg17_12")
fn.load.data("xg17_13")
fn.load.data("xg17_14")
###############################################################################################
pred <- TGT
pred[ , 1 ] <- xg17_1$train
pred[ , 2 ] <- xg17_2$train
pred[ , 3 ] <- xg17_3$train
pred[ , 4 ] <- xg17_4$train
pred[ , 5 ] <- xg17_5$train 
pred[ , 6 ] <- xg17_6$train
pred[ , 7 ] <- xg17_7$train
pred[ , 8 ] <- xg17_8$train
pred[ , 9 ] <- xg17_9$train
pred[ , 10 ] <- xg17_10$train
pred[ , 11 ] <- xg17_11$train
pred[ , 12 ] <- xg17_12$train
pred[ , 13 ] <- xg17_13$train
pred[ , 14 ] <- xg17_14$train
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
# 0.2501834
# [1] 0.3266423
# [1] 0.4758945
# [1] 0.4931855
# [1] 0.02754607
# [1] 0.1544483
# [1] 0.06047558
# [1] 0.1667512
# [1] 0.4300604
# [1] 0.05404174
# [1] 0.3587536
# [1] 0.4553538
# [1] 0.08685779
# [1] 0.08084391
# [1] 0.3317125


sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub$service_a <- xg17_1$test
sub$service_b <- xg17_2$test
sub$service_c <- xg17_3$test
sub$service_d <- xg17_4$test
sub$service_e <- xg17_5$test
sub$service_f <- xg17_6$test
sub$service_g <- xg17_7$test
sub$service_h <- xg17_8$test
sub$service_i <- xg17_9$test
sub$service_j <- xg17_10$test
sub$service_k <- xg17_11$test
sub$service_l <- xg17_12$test
sub$service_m <- xg17_13$test
sub$service_n <- xg17_14$test

pred.xg.17 <- list(  train=pred , test=as.matrix(sub[ , 2:ncol(sub)  ]) )
fn.save.data("pred.xg.17")
str( pred.xg.17$train  )
str( pred.xg.17$test  )

