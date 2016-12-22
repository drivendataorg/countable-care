rm(list = ls(all = TRUE))
source("fn.base.R")
library(xgboost)
library(data.table)
library(foreach)
gc()

fn.load.data("train")
fn.load.data("test")
target <- fread( 'input/train_labels.csv' )
TGT <- as.matrix( target )
TGT <- TGT[,2:ncol(TGT)]


#Remove feature levels that appear only once
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
fn.load.data("TAB")


cs <- colSums(train < -90)
cs <- order(cs)
train <- train[ , cs]
test  <- test[ , cs]
gc()

l <- lapply(  lapply( as.data.frame(train),unique ) , length )
train <- train[ , which(l>1) ]
test  <- test[  , which(l>1) ]

cv <- rep( 1:4 , length.out=nrow(train)   )

FEAT = 1
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1300 ,shri=0.0055, depth=10,subsample=0.750,colsample=0.50, verbose=TRUE, trainFULL=TRUE )
feats <- 5:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1300 ,shri=0.0055, depth=11,subsample=0.750,colsample=0.50, verbose=TRUE, trainFULL=TRUE )
feats <- 10:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1300 ,shri=0.0055, depth=12,subsample=0.750,colsample=0.50, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_1 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_1")

FEAT = 2
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0085, depth=5,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
feats <- 10:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0085, depth=5,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
feats <- 5:ncol(train)
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0085, depth=7,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_2 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_2")

FEAT = 3
feats <- 1:1345
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0085, depth=7,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
feats <- 10:1355
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0085, depth=7,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
feats <- 20:ncol(train)
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0085, depth=7,subsample=0.750,colsample=0.2750, verbose=FALSE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_3 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_3")

FEAT = 4
feats <- 15:1350
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.2750, verbose=TRUE, trainFULL=TRUE )
feats <- 10:1345
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.850,colsample=0.300, verbose=TRUE, trainFULL=TRUE )
feats <- 15:1355
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.80,colsample=0.250, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_4 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_4")

FEAT = 5
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=5,subsample=0.750,colsample=0.2750, verbose=TRUE, trainFULL=TRUE )
feats <- 5:660
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=6,subsample=0.750,colsample=0.2750, verbose=TRUE, trainFULL=TRUE )
feats <- 10:770
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=7,subsample=0.750,colsample=0.2750, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_5 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_5")

FEAT = 6
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 5:660
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=5,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 10:770
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=5,subsample=0.7750,colsample=0.33, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_6 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_6")

FEAT = 7
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 5:660
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=5,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 10:770
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=5,subsample=0.7750,colsample=0.33, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_7 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_7")

FEAT = 8
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 5:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.7750,colsample=0.33, verbose=TRUE, trainFULL=TRUE )
feats <- 10:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.80,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_8 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_8")

FEAT = 9
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 5:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.7750,colsample=0.33, verbose=TRUE, trainFULL=TRUE )
feats <- 10:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.80,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_9 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_9")

FEAT = 10
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 10:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 20:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=5,subsample=0.80,colsample=0.350, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_10 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_10")

FEAT = 11
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 10:560
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=4,subsample=0.750,colsample=0.30, verbose=TRUE, trainFULL=TRUE )
feats <- 20:570
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.009, depth=5,subsample=0.80,colsample=0.350, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_11 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_11")

FEAT = 12
feats <- 1:650
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0145, depth=8,subsample=0.750,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
feats <- 5:1000
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0145, depth=8,subsample=0.750,colsample=0.45, verbose=TRUE, trainFULL=TRUE )
feats <- 10:750
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0145, depth=8,subsample=0.80,colsample=0.40, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_12 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_12")

FEAT = 13
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0145, depth=8,subsample=0.750,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
feats <- 5:650
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0145, depth=8,subsample=0.750,colsample=0.45, verbose=TRUE, trainFULL=TRUE )
feats <- 10:750
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.0145, depth=8,subsample=0.80,colsample=0.40, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_13 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_13")

FEAT = 14
feats <- 1:550
xgA <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.007, depth=8,subsample=0.750,colsample=0.35, verbose=TRUE, trainFULL=TRUE )
feats <- 5:650
xgB <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.007, depth=8,subsample=0.750,colsample=0.45, verbose=TRUE, trainFULL=TRUE )
feats <- 10:750
xgC <- xgbCV2( train[,feats] , test[,feats] , TGT[,FEAT],  cv, ite=1200 ,shri=0.007, depth=8,subsample=0.80,colsample=0.40, verbose=TRUE, trainFULL=TRUE )
print(llfun(  TGT[,FEAT] ,   xgA$train    ))
print(llfun(  TGT[,FEAT] ,   xgB$train    ))
print(llfun(  TGT[,FEAT] ,   xgC$train    ))
print(llfun(  TGT[,FEAT] ,   (xgA$train     + xgB$train     + xgC$train)/3  ))
xg12_14 <- list( train = (xgA$train     + xgB$train     + xgC$train)/3 ,
               test  = (xgA$testFULL  + xgB$testFULL  + xgC$testFULL )/3  )
fn.save.data("xg12_14")

fn.load.data("xg12_1")
fn.load.data("xg12_2")
fn.load.data("xg12_3")
fn.load.data("xg12_4")
fn.load.data("xg12_5")
fn.load.data("xg12_6")
fn.load.data("xg12_7")
fn.load.data("xg12_8")
fn.load.data("xg12_9")
fn.load.data("xg12_10")
fn.load.data("xg12_11")
fn.load.data("xg12_12")
fn.load.data("xg12_13")
fn.load.data("xg12_14")

pred <- TGT
pred[ , 1 ] <- xg12_1$train
pred[ , 2 ] <- xg12_2$train
pred[ , 3 ] <- xg12_3$train
pred[ , 4 ] <- xg12_4$train
pred[ , 5 ] <- xg12_5$train
pred[ , 6 ] <- xg12_6$train
pred[ , 7 ] <- xg12_7$train
pred[ , 8 ] <- xg12_8$train
pred[ , 9 ] <- xg12_9$train
pred[ , 10 ] <- xg12_10$train
pred[ , 11 ] <- xg12_11$train
pred[ , 12 ] <- xg12_12$train
pred[ , 13 ] <- xg12_13$train
pred[ , 14 ] <- xg12_14$train
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
#0.2510743
# [1] 0.3315252
# [1] 0.482824
# [1] 0.4939136
# [1] 0.02770036
# [1] 0.1542076
# [1] 0.05923756
# [1] 0.167108
# [1] 0.4335903
# [1] 0.05166877
# [1] 0.3609043
# [1] 0.4529225
# [1] 0.08559111
# [1] 0.07779479
# [1] 0.3360524

sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub$service_a <- xg12_1$test
sub$service_b <- xg12_2$test
sub$service_c <- xg12_3$test
sub$service_d <- xg12_4$test
sub$service_e <- xg12_5$test
sub$service_f <- xg12_6$test
sub$service_g <- xg12_7$test
sub$service_h <- xg12_8$test
sub$service_i <- xg12_9$test
sub$service_j <- xg12_10$test
sub$service_k <- xg12_11$test
sub$service_l <- xg12_12$test
sub$service_m <- xg12_13$test
sub$service_n <- xg12_14$test

pred.xg.12 <- list(  train=pred , test=as.matrix(sub[ , 2:ncol(sub)  ]) )
fn.save.data("pred.xg.12")
str( pred.xg.12$train  )
str( pred.xg.12$test  )

# write.table( sub , "submission/sub12.csv", row.names=FALSE, quote=FALSE,sep="," )
