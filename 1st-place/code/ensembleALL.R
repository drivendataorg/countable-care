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
set.seed(11111)



#Load Target
target <- fread( 'input/train_labels.csv' )
TGT <- as.matrix( target )
TGT <- TGT[,2:ncol(TGT)]
######################################################################################################
######################################################################################################
######################################################################################################
#




#
######################################################################################################
######################################################################################################
######################################################################################################
#Load ALL Models
fn.load.data("train")
fn.load.data("test")
fn.load.data("pred.rf.10")
fn.load.data("pred.xg.9")
fn.load.data("pred.xg.12")
fn.load.data("pred.xg.13")
fn.load.data("pred.xg.17")
fn.load.data("pred.xg.19")
fn.load.data("pred.xg.20")

pred.train1 <- matrix( 0  , nrow=nrow(train) , ncol=14  )
for( i in 0:13){
  pred <- fread( paste0('predictions/predtrain1_',i,'.csv') , header=FALSE  )
  pred.train1[,i+1] <- pred$V1
  print( paste( i , llfun( TGT[,i+1] , pred$V1  ) ) )
}
pred.train2 <- matrix( 0  , nrow=nrow(train) , ncol=14  )
for( i in 0:13){
  pred <- fread( paste0('predictions/predtrain2_',i,'.csv') , header=FALSE  )
  pred.train2[,i+1] <- pred$V1
  print( paste( i , llfun( TGT[,i+1] , pred$V1  ) ) )
}

pred.test1 <- matrix( 0  , nrow=nrow(test) , ncol=14  )
for( i in 0:13){
  pred <- fread( paste0('predictions/predtest1_',i,'.csv') , header=FALSE  )
  pred.test1[,i+1] <- pred$V1
}

pred.test2 <- matrix( 0  , nrow=nrow(test) , ncol=14  )
for( i in 0:13){
  pred <- fread( paste0('predictions/predtest2_',i,'.csv') , header=FALSE  )
  pred.test2[,i+1] <- pred$V1
}

#Test Models
LogLoss( TGT , pred.rf.10$train )
LogLoss( TGT , pred.xg.9$train )
LogLoss( TGT , pred.xg.12$train )
LogLoss( TGT , pred.xg.13$train )
LogLoss( TGT , pred.xg.17$train )
LogLoss( TGT , pred.xg.19$train )
LogLoss( TGT , pred.xg.20$train )
LogLoss( TGT , pred.train1 )
LogLoss( TGT , pred.train2 )
######################################################################################################
######################################################################################################
######################################################################################################
#




#
######################################################################################################
#ENSEMBLE MODEL 1#####################################################################################
######################################################################################################
pred.train <- matrix(0,nrow=nrow(train),ncol=14)
pred.test  <- matrix(0,nrow=nrow(test),ncol=14);gc()
s=0
FEAT=1
for( FEAT in 1:14  ){
  tgt <- TGT[,FEAT]
  tr <- cbind( pred.xg.9$train, pred.rf.10$train, pred.train1, pred.train2,  pred.xg.12$train,  pred.xg.13$train,  pred.xg.17$train,  pred.xg.19$train,  pred.xg.20$train  )
  ts <- cbind( pred.xg.9$test , pred.rf.10$test , pred.test1 , pred.test2 ,  pred.xg.12$test ,  pred.xg.13$test ,  pred.xg.17$test ,  pred.xg.19$test ,  pred.xg.20$test   )  
  xg <- xgbCV2( tr , ts , tgt,  cv,
                ite=5000 ,shri=0.0014, depth=3,subsample=0.60,colsample=0.40, verbose=FALSE, trainFULL=FALSE, ncores=7)
  print( paste( FEAT,llfun( tgt , xg$train  ) ) )
  
  pred.train[,FEAT] <- xg$train
  pred.test[,FEAT]  <- xg$test
  gc()  
}
LogLoss( TGT , pred.train   )
# [1] "1 0.322251267268813"
# [1] "2 0.470462918190159"
# [1] "3 0.485882935717789"
# [1] "4 0.0264986927299437"
# [1] "5 0.148439646875084"
# [1] "6 0.0603920507535576"
# [1] "7 0.164929324139826"
# [1] "8 0.428799701356267"
# [1] "9 0.0476108814493492"
# [1] "10 0.354903050817818"
# [1] "11 0.448121519989039"
# [1] "12 0.0829882968485582"
# [1] "13 0.0755232275692476"
# [1] "14 0.333495686100297"
# [1] 0.2464499
sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub[,2:15] <- pred.test
summary(sub)
write.table( sub , "submission/sub.ok.1.csv", row.names=FALSE, quote=FALSE,sep="," )

sub.ok.1 <- list( train=pred.train, test=pred.test )
fn.save.data("sub.ok.1")
######################################################################################################
#




#
######################################################################################################
######################################################################################################
#ENSEMBLE MODEL 2#####################################################################################
######################################################################################################
######################################################################################################
pred.train <- matrix(0,nrow=nrow(train),ncol=14)
pred.test  <- matrix(0,nrow=nrow(test),ncol=14);gc()
s=0
FEAT=1
for( FEAT in 1:14  ){
  tgt <- TGT[,FEAT]
  tr <- cbind( pred.rf.10$train, pred.train1, pred.xg.12$train,  pred.xg.13$train,  pred.xg.17$train,  pred.xg.19$train)
  ts <- cbind( pred.rf.10$test , pred.test1 , pred.xg.12$test ,  pred.xg.13$test ,  pred.xg.17$test ,  pred.xg.19$test )
  xg <- xgbCV2( tr , ts , tgt,  cv,
                ite=5000 ,shri=0.0014, depth=3,subsample=0.60,colsample=0.40, verbose=FALSE, trainFULL=FALSE, ncores=7)
  print( paste( FEAT,llfun( tgt , xg$train  ) ) )
  
  pred.train[,FEAT] <- xg$train
  pred.test[,FEAT]  <- xg$test
  gc()  
}
LogLoss( TGT , pred.train   )
# [1] "1 0.322638855695475"
# [1] "2 0.470235842687603"
# [1] "3 0.485980275145513"
# [1] "4 0.0261543170044054"
# [1] "5 0.148447316958147"
# [1] "6 0.0603504700853337"
# [1] "7 0.165046889114658"
# [1] "8 0.428898295574351"
# [1] "9 0.0474058897635826"
# [1] "10 0.354673799991112"
# [1] "11 0.447960454529191"
# [1] "12 0.0830447279495161"
# [1] "13 0.0753050368283715"
# [1] "14 0.333542132985541"
# [1] 0.246406
sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub[,2:15] <- pred.test
summary(sub)
write.table( sub , "submission/sub.ok.2.csv", row.names=FALSE, quote=FALSE,sep="," )

sub.ok.2 <- list( train=pred.train, test=pred.test )
fn.save.data("sub.ok.2")
######################################################################################################
#




#
######################################################################################################
######################################################################################################
#ENSEMBLE MODEL 3#####################################################################################
######################################################################################################
######################################################################################################
pred.train <- matrix(0,nrow=nrow(train),ncol=14)
pred.test  <- matrix(0,nrow=nrow(test),ncol=14);gc()
s=0
FEAT=1
for( FEAT in 1:14  ){
  tgt <- TGT[,FEAT]
  tr <- cbind( pred.rf.10$train, pred.train1, pred.xg.13$train,  pred.xg.17$train )
  ts <- cbind( pred.rf.10$test , pred.test1 , pred.xg.13$test ,  pred.xg.17$test  )
  xg <- xgbCV2( tr , ts , tgt,  cv,
                ite=5000 ,shri=0.0014, depth=3,subsample=0.60,colsample=0.40, verbose=FALSE, trainFULL=FALSE, ncores=7)
  print( paste( FEAT,llfun( tgt , xg$train  ) ) )
  
  pred.train[,FEAT] <- xg$train
  pred.test[,FEAT]  <- xg$test
  gc()  
}
LogLoss( TGT , pred.train   )
# [1] "1 0.324278127101967"
# [1] "2 0.470920749471809"
# [1] "3 0.486561538070861"
# [1] "4 0.0262784502576811"
# [1] "5 0.148960102786008"
# [1] "6 0.0606142562268661"
# [1] "7 0.165054146756839"
# [1] "8 0.429409078672329"
# [1] "9 0.0476144846480191"
# [1] "10 0.354762058348397"
# [1] "11 0.44802457245272"
# [1] "12 0.0836023158161042"
# [1] "13 0.0760556777065902"
# [1] "14 0.333844458484916"
# [1] 0.2468557
sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub[,2:15] <- pred.test
summary(sub)
write.table( sub , "submission/sub.ok.3.csv", row.names=FALSE, quote=FALSE,sep="," )

sub.ok.3 <- list( train=pred.train, test=pred.test )
fn.save.data("sub.ok.3")
######################################################################################################
#





#
######################################################################################################
fn.load.data("sub.ok.1")
fn.load.data("sub.ok.2")
fn.load.data("sub.ok.3")
LogLoss( TGT , (sub.ok.1$train + sub.ok.2$train + sub.ok.3$train)/3  )


sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub[,2:15] <- (sub.ok.1$test + sub.ok.2$test + sub.ok.3$test)/3  # FINAL BAG
summary(sub)
write.table( sub , "submission/final_aritmetic_ensemble.csv", row.names=FALSE, quote=FALSE,sep="," )
######################################################################################################

