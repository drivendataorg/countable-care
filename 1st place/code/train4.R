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


#Linearize Categorical Features
tic()
fold=2
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,1] , cv  );gc()
  str <- llfun( TGT[,1] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
toc()

tr1 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts1 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
i=1
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr1[,n] <- trainpar[[i]] 
  ts1[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr1")
fn.save.data("ts1")


tic()
fold=2
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,2] , cv  );gc()
  str <- llfun( TGT[,2] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug2.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
toc()

tr2 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts2 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
i=1
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr2[,n] <- trainpar[[i]] 
  ts2[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr2")
fn.save.data("ts2")


tic()
fold=2
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,3] , cv  );gc()
  str <- llfun( TGT[,3] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug3.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
toc()

tr3 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts3 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
i=1
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr3[,n] <- trainpar[[i]] 
  ts3[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr3")
fn.save.data("ts3")

tic()
fold=2
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,8] , cv  );gc()
  str <- llfun( TGT[,8] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug8.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
toc()

tr8 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts8 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
i=1
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr8[,n] <- trainpar[[i]] 
  ts8[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr8")
fn.save.data("ts8")


tic()
fold=2
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,11] , cv  );gc()
  str <- llfun( TGT[,11] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug11.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
toc()

tr11 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts11 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
i=1
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr11[,n] <- trainpar[[i]] 
  ts11[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr11")
fn.save.data("ts11")



tic()
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,4] , cv  );gc()
  str <- llfun( TGT[,4] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug4.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
toc()

tr4 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts4 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
i=1
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr4[,n] <- trainpar[[i]] 
  ts4[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr4")
fn.save.data("ts4")



tic()
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,5] , cv  );gc()
  str <- llfun( TGT[,5] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug5.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
toc()

tr5 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts5 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
i=1
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr5[,n] <- trainpar[[i]] 
  ts5[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr5")
fn.save.data("ts5")



tic()
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  print(fold)
  if( fold<326 ){
    TR <- data.frame( A=c(train[,fold],test[,fold] ),B=c(train[,fold],test[,fold] )^2 )
    TR <- scale(TR)
    TR[ is.na(TR) ] <- mean( TR,na.rm=TRUE  )
    TR <- as.data.frame(TR)
  }else{
    TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
    TR <- data.frame(model.matrix(~.-1,TR))
  }
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,6] , cv  );gc()
  str <- llfun( TGT[,6] , g1$train)
  cat( paste(fold,str,ncol(TR),'\n') , file=paste0('debug/debug6.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
tr6 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts6 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr6[,n] <- trainpar[[i]] 
  ts6[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr6")
fn.save.data("ts6")
toc()



tic()
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,7] , cv  );gc()
  str <- llfun( TGT[,7] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug7.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
tr7 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts7 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr7[,n] <- trainpar[[i]] 
  ts7[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr7")
fn.save.data("ts7")
toc()



tic()
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,9] , cv  );gc()
  str <- llfun( TGT[,9] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug9.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
tr9 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts9 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr9[,n] <- trainpar[[i]] 
  ts9[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr9")
fn.save.data("ts9")
toc()




tic()
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,10] , cv  );gc()
  str <- llfun( TGT[,10] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug10.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
tr10 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts10 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr10[,n] <- trainpar[[i]] 
  ts10[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr10")
fn.save.data("ts10")
toc()




tic()
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,12] , cv  );gc()
  str <- llfun( TGT[,12] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug12.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
tr12 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts12 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr12[,n] <- trainpar[[i]] 
  ts12[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr12")
fn.save.data("ts12")
toc()



tic()
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,13] , cv  );gc()
  str <- llfun( TGT[,13] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug13.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
tr13 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts13 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr13[,n] <- trainpar[[i]] 
  ts13[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr13")
fn.save.data("ts13")
toc()



tic()
fn.register.wk(8)
trainpar <- foreach(fold=2:ncol(train), .combine=c) %dopar% {
  TR <- data.frame( A=factor( c(train[,fold],test[,fold]) ) )
  TR <- data.frame(model.matrix(~.-1,TR))
  g1 <- glmFACTOR( TR[px,], TR[py,], TGT[,14] , cv  );gc()
  str <- llfun( TGT[,14] , g1$train)
  cat( paste(fold,str,'\n') , file=paste0('debug/debug14.txt'), sep="", append=TRUE)
  list( train=g1$train , test=g1$test )
}
fn.kill.wk()
tr14 <- matrix( 0  ,nrow=nrow(train) , ncol=ncol(train) )
ts14 <- matrix( 0  ,nrow=nrow(test) , ncol=ncol(test) )
n=1
for( i in seq(1,length(trainpar),2)  ){
  tr14[,n] <- trainpar[[i]] 
  ts14[,n] <- trainpar[[i+1]]
  n=n+1
}
fn.save.data("tr14")
fn.save.data("ts14")
toc()





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

fn.load.data("xg13_1")
llfun( TGT[,tgt] , xg13_1$train  )
#
cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- c( 1 , which( sc < 1.00*median(sc[2:ncol(train)]) ) )
feats <- 1:850
xg13_1 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv,
                  ite=15000 ,shri=0.0010, depth=6,subsample=0.75,colsample=0.25, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_1$train  )
fn.save.data("xg13_1")
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
feats <- 1:200
xg13_2 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1500 ,shri=0.0090, depth=6,subsample=0.750,colsample=0.250, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_2$train  )
fn.save.data("xg13_2")
#
#
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
feats <- 1:750
xg13_3 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=2500 ,shri=0.015, depth=5,subsample=0.75,colsample=0.15, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_3$train  )
fn.save.data("xg13_3")
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
feats <- c( 1 , which( sc < 1.00*median(sc[2:ncol(train)]) ) )
feats <- 1:200
feats <- c( 1 , which( sc < 1.00*median(sc[2:ncol(train)]) ) )
xg13_8 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1500 ,shri=0.012, depth=6,subsample=0.750,colsample=0.250, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_8$train  )
fn.save.data("xg13_8")
#
#
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
feats <- c( 1 , which( sc < 1.00*median(sc[2:ncol(train)]) ) )
feats <- 1:200
xg13_11 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1500 ,shri=0.012, depth=6,subsample=0.750,colsample=0.250, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_11$train  )
fn.save.data("xg13_11")


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
feats <- c( 1 , which( sc < 1.00*median(sc[2:ncol(train)]) ) )
feats <- 1:200
xg13_4 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=1500 ,shri=0.011, depth=3,subsample=0.750,colsample=0.200, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_4$train  )
fn.save.data("xg13_4")
#
#
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
feats <- c( 1 , which( sc < 0.995*median(sc[2:ncol(train)]) ) )
feats <- 1:200
xg13_5 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=2000 ,shri=0.015, depth=3,subsample=0.75,colsample=0.200, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_5$train  )
fn.save.data("xg13_5")
#
#
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
xg13_6 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=2500 ,shri=0.005, depth=3,subsample=0.750,colsample=0.250, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_6$train  )
fn.save.data("xg13_6")
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

fn.load.data("xg13_7")
llfun( TGT[,tgt] , xg13_7$train  )
cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- c( 1 , which( sc < 0.995*median(sc[2:ncol(train)]) ) )
feats <- 1:600
xg13_7 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=2500 ,shri=0.0075, depth=5,subsample=0.750,colsample=0.250, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_7$train  )
fn.save.data("xg13_7")
#
#
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
feats <- c( 1 , which( sc < 0.995*median(sc[2:ncol(train)]) ) )
feats <- 1:450
xg13_9 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=2500 ,shri=0.005, depth=3,subsample=0.750,colsample=0.250, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_9$train  )
fn.save.data("xg13_9")
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

fn.load.data("xg13_10")
llfun( TGT[,tgt] , xg13_10$train  )
cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- c( 1 , which( sc < 0.995*median(sc[2:ncol(train)]) ) )
feats <- 1:1250
xg13_10 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                  ite=2500 ,shri=0.0135, depth=3,subsample=0.75,colsample=0.25, verbose=FALSE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_10$train  )
fn.save.data("xg13_10")
#
#
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

fn.load.data("xg13_12")
llfun( TGT[,tgt] , xg13_12$train  )
cv <- rep( 1:4 , length.out=nrow(train)   )
feats <- c( 1 , which( sc < 0.995*median(sc[2:ncol(train)]) ) )
feats <- 1:750
xg13_12 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                   ite=2500 ,shri=0.010, depth=3,subsample=0.750,colsample=0.250, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_12$train  )
fn.save.data("xg13_12")
#
#
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
feats <- c( 1 , which( sc < 0.995*median(sc[2:ncol(train)]) ) )
feats <- 1:1250
xg13_13 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                   ite=2500 ,shri=0.0135, depth=3,subsample=0.750,colsample=0.250, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_13$train  )
fn.save.data("xg13_13")
#
#
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
feats <- c( 1 , which( sc < 0.995*median(sc[2:ncol(train)]) ) )
feats <- 1:1250
xg13_14 <- xgbCV2( train[,feats] , test[,feats] , TGT[,tgt],  cv, 
                   ite=2500 ,shri=0.0135, depth=3,subsample=0.750,colsample=0.250, verbose=TRUE, trainFULL=TRUE, ncores=7 )
llfun( TGT[,tgt] , xg13_14$train  )
fn.save.data("xg13_14")
#
#
#
############################################################################
############################################################################
############################################################################




############################################################################
fn.load.data("train")
fn.load.data("test")
target <- fread( 'input/train_labels.csv' )
TGT <- as.matrix( target )
TGT <- TGT[,2:ncol(TGT)]

TAB <- data.frame( i=0, n=0, f=0   )
i=181
for( i in 1:ncol(train)  ){
  tmp <- c( train[,i] , test[,i] )
  tab <- data.frame( table(tmp)  )
  tab <- tab[ order(tab$Freq,decreasing=TRUE)  ,]
  tab$tmp <- as.numeric( as.character(tab$tmp)  )
  tab <- tab[ tab$Freq==1 ,]
  tmp[ tmp %in% tab$tmp  ] <- -99
  
  print( paste( i , nrow(tab) , sum(tab$Freq==1) )  )
  TAB <- rbind( TAB , data.frame( i=i , n=nrow(tab) , f=sum(tab$Freq==1) )  )
  train[,i] <- tmp[ 1:nrow(train) ]
  test[,i]  <- tmp[ (nrow(train)+1):length(tmp) ]
}
fn.save.data("TAB")

cs <- colSums(train < -90)
cs <- order(cs)
train <- train[ , cs]
test  <- test[ , cs]
l <- lapply(  lapply( as.data.frame(train),unique ) , length )
train <- train[ , which(l>1) ]
test  <- test[  , which(l>1) ]
gc()
cv <- rep( 1:4 , length.out=nrow(train)   )

FEAT = 9
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE, ncores=4 )
feats <- 5:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.7750,colsample=0.33, verbose=TRUE, trainFULL=TRUE, ncores=4 )
feats <- 10:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.80,colsample=0.35, verbose=TRUE, trainFULL=TRUE, ncores=4 )
print(llfun(  TGT[,FEAT] ,   xgA$train    ));gc()
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_9 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
                test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_9")


###############################################################################################
###############################################################################################
###############################################################################################
###############################################################################################
fn.load.data("xg13_1")
fn.load.data("xg13_2")
fn.load.data("xg13_3")
fn.load.data("xg13_4")
fn.load.data("xg13_5")
fn.load.data("xg13_6")
fn.load.data("xg13_7")
fn.load.data("xg13_8")
fn.load.data("xg13_9")
fn.load.data("xg13_10")
fn.load.data("xg13_11")
fn.load.data("xg13_12")
fn.load.data("xg13_13")
fn.load.data("xg13_14")
###############################################################################################
pred <- TGT
pred[ , 1 ] <- xg13_1$train
pred[ , 2 ] <- xg13_2$train
pred[ , 3 ] <- xg13_3$train
pred[ , 4 ] <- xg13_4$train
pred[ , 5 ] <- xg13_5$train 
pred[ , 6 ] <- xg13_6$train
pred[ , 7 ] <- xg13_7$train
pred[ , 8 ] <- xg13_8$train
pred[ , 9 ] <- xg13_9$train
pred[ , 10 ] <- xg13_10$train
pred[ , 11 ] <- xg13_11$train
pred[ , 12 ] <- xg13_12$train
pred[ , 13 ] <- xg13_13$train
pred[ , 14 ] <- xg13_14$train
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
# 0.2487473
# [1] 0.326032
# [1] 0.4768665
# [1] 0.4914302
# [1] 0.02689491
# [1] 0.1505364
# [1] 0.05849888
# [1] 0.1656084
# [1] 0.4305456
# [1] 0.05166877
# [1] 0.3542565
# [1] 0.4547467
# [1] 0.08362942
# [1] 0.0794038
# [1] 0.3323447


sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub$service_a <- xg13_1$test
sub$service_b <- xg13_2$test
sub$service_c <- xg13_3$test
sub$service_d <- xg13_4$test
sub$service_e <- xg13_5$test
sub$service_f <- xg13_6$test
sub$service_g <- xg13_7$test
sub$service_h <- xg13_8$test
sub$service_i <- xg13_9$test
sub$service_j <- xg13_10$test
sub$service_k <- xg13_11$test
sub$service_l <- xg13_12$test
sub$service_m <- xg13_13$test
sub$service_n <- xg13_14$test


pred.xg.13 <- list(  train=pred , test=as.matrix(sub[ , 2:ncol(sub)  ]) )
fn.save.data("pred.xg.13")
str( pred.xg.13$train  )
str( pred.xg.13$test  )

# write.table( sub , "submission/sub13A.csv", row.names=FALSE, quote=FALSE,sep="," )





