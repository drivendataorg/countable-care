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
feats <- 1:650
xg19_1 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv,
                  ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_1$train  )
fn.save.data("xg19_1")
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
feats <- 1:650
xg19_2 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_2$train  )
fn.save.data("xg19_2")
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
feats <- 1:650
xg19_3 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_3$train  )
fn.save.data("xg19_3")
# [1] "Searching best tree number..."
# [1] "847 0.491100371585719"
#
#
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
feats <- 1:650
xg19_8 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_8$train  )
fn.save.data("xg19_8")
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
feats <- 1:650
xg19_11 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_11$train  )
fn.save.data("xg19_11")


#
#
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
feats <- 1:650
xg19_4 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_4$train  )
fn.save.data("xg19_4")
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
feats <- 1:650
xg19_5 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_5$train  )
fn.save.data("xg19_5")
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
feats <- c( 1 , which( sc < 0.995*median(sc[2:ncol(train)]) ) )
feats <- 1:450
feats <- 1:650
xg19_6 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
#                   ite=5000 ,shri=0.00255, depth=3,subsample=0.750,colsample=0.300, verbose=TRUE, trainFULL=TRUE, ncores=7 )
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_6$train  )
fn.save.data("xg19_6")
#
#
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
feats <- 1:650
xg19_7 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_7$train  )
fn.save.data("xg19_7")
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
feats <- 1:650
xg19_9 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_9$train  )
fn.save.data("xg19_9")
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
feats <- 1:650
xg19_10 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_10$train  )
fn.save.data("xg19_10")
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
feats <- 1:650
xg19_12 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_12$train  )
fn.save.data("xg19_12")
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
feats <- 1:650
xg19_13 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_13$train  )
fn.save.data("xg19_13")
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
feats <- 1:650
xg19_14 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
#                    ite=5000 ,shri=0.007135, depth=3,subsample=0.80,colsample=0.300, verbose=TRUE, trainFULL=TRUE, ncores=7 )
                    ite=1000 ,shri=0.018, depth=6,subsample=0.50,colsample=0.50, verbose=TRUE, trainFULL=TRUE, ncores=7)
llfun( TGT[,tgt] , xg19_14$train  )
fn.save.data("xg19_14")


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
fn.load.data("xg19_1")
fn.load.data("xg19_2")
fn.load.data("xg19_3")
fn.load.data("xg19_4")
fn.load.data("xg19_5")
fn.load.data("xg19_6")
fn.load.data("xg19_7")
fn.load.data("xg19_8")
fn.load.data("xg19_9")
fn.load.data("xg19_10")
fn.load.data("xg19_11")
fn.load.data("xg19_12")
fn.load.data("xg19_13")
fn.load.data("xg19_14")
###############################################################################################
pred <- TGT
pred[ , 1 ] <- xg19_1$train
pred[ , 2 ] <- xg19_2$train
pred[ , 3 ] <- xg19_3$train
pred[ , 4 ] <- xg19_4$train
pred[ , 5 ] <- xg19_5$train 
pred[ , 6 ] <- xg19_6$train
pred[ , 7 ] <- xg19_7$train
pred[ , 8 ] <- xg19_8$train
pred[ , 9 ] <- xg19_9$train
pred[ , 10 ] <- xg19_10$train
pred[ , 11 ] <- xg19_11$train
pred[ , 12 ] <- xg19_12$train
pred[ , 13 ] <- xg19_13$train
pred[ , 14 ] <- xg19_14$train
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
# 0.251123
# [1] 0.3302426
# [1] 0.4753689
# [1] 0.4930464
# [1] 0.02802717
# [1] 0.153755
# [1] 0.06319045
# [1] 0.1685062
# [1] 0.4337626
# [1] 0.05653982
# [1] 0.3589537
# [1] 0.4549053
# [1] 0.08480564
# [1] 0.08067409
# [1] 0.3339441


sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub$service_a <- xg19_1$test
sub$service_b <- xg19_2$test
sub$service_c <- xg19_3$test
sub$service_d <- xg19_4$test
sub$service_e <- xg19_5$test
sub$service_f <- xg19_6$test
sub$service_g <- xg19_7$test
sub$service_h <- xg19_8$test
sub$service_i <- xg19_9$test
sub$service_j <- xg19_10$test
sub$service_k <- xg19_11$test
sub$service_l <- xg19_12$test
sub$service_m <- xg19_13$test
sub$service_n <- xg19_14$test

pred.xg.19 <- list(  train=pred , test=as.matrix(sub[ , 2:ncol(sub)  ]) )
fn.save.data("pred.xg.19")
str( pred.xg.19$train  )
str( pred.xg.19$test  )

# write.table( sub , "submission/sub19A.csv", row.names=FALSE, quote=FALSE,sep="," )
