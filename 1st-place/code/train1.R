rm(list = ls(all = TRUE))
source("fn.base.R")
library(xgboost)
library(data.table)
library(randomForest)
# library(e1071)
gc()
set.seed(1)

###############################################################3
train <- fread( 'input/train_values.csv', stringsAsFactors=TRUE, colClasses="character" )
test  <- fread( 'input/test_values.csv', stringsAsFactors=TRUE, colClasses="character"  )
target <- fread( 'input/train_labels.csv' )
str(train)

train[  train==""  ] <- NA
test[  test==""  ] <- NA
train <- as.matrix(train)
test <- as.matrix(test);gc()

train <- train[,2:ncol(train)]
test  <- test[,2:ncol(test)]

SNA <- rowSums( is.na(train)   )
train <- cbind( SNA , train )
SNA <- rowSums( is.na(test)   )
test <- cbind( SNA , test )
gc()

for( i in 3:118  ){#process NAs
  print(i)
  v <- as.numeric( c( train[,i],test[,i]  )  )
  v[ is.na(v) ] <- -999
  train[,i] <- v[1:nrow(train)]
  test[,i] <- v[(nrow(train)+1):length(v)]
}

for( i in 119:329  ){#process NAs
  print(i)
  v <- as.numeric( c( train[,i],test[,i]  ) )
  v[ is.na(v) ] <- -999
  train[,i] <- v[1:nrow(train)]
  test[,i] <- v[(nrow(train)+1):length(v)]
}

for( i in c(2,330:ncol(train))  ){#process categorical NAs
  print(i)
  v <- as.numeric( factor(c( train[,i],test[,i]  ) ) )
  v[ is.na(v) ] <- -999
  train[,i] <- v[1:nrow(train)]
  test[,i] <- v[(nrow(train)+1):length(v)]  
};gc()
train <- matrix( as.numeric(train) , nrow=nrow(train) , ncol=ncol(train) )
test  <- matrix( as.numeric(test)  , nrow=nrow(test)  , ncol=ncol(test) )

target <- fread( 'input/train_labels.csv' )
str(train)
TGT <- as.matrix( target )
TGT <- TGT[,2:ncol(TGT)]
TGTi <- max.col(TGT)-1
table(TGTi)
# TGTi
# 1    2    3    4    5    6    7    8    9   10   11   12   13   14 
# 1945 1106  921   43  133   95  153 1066   45 4204 3719  335  235  644 


#sort feature from columns with small number of NAs to columns with much NAs.
cs <- colSums(train == -999)
cs <- order(cs)
train <- train[ , cs]
test  <- test[ , cs]
gc()

train <- train[,1:1365]
test <- test[,1:1365]

#Save preprocessed dataset to python use later
write.table( train , "input/train.csv", row.names=FALSE, col.names=FALSE, quote=FALSE,sep="," )
write.table( test  , "input/test.csv", row.names=FALSE, col.names=FALSE, quote=FALSE,sep="," )
write.table( TGT   , "input/target.csv", row.names=FALSE, col.names=FALSE, quote=FALSE,sep="," )

fn.save.data("train")
fn.save.data("test")
fn.save.data("TGT")
fn.save.data("TGTi")
###############################################################



###############################################################
fn.load.data("train")
fn.load.data("test")
fn.load.data("TGT")
fn.load.data("TGTi")


# Compute Random Forest Feature Importance for all 14 targets independently
fn.register.wk(7)
RFimp <- foreach(fold=1:14, .combine=cbind, .inorder=TRUE, .packages = "randomForest") %dopar% { # compute importance for all
  rf <-  randomForest( x=train[,1:ncol(train)], y=factor(TGT[,fold]), ntree=400, importance=TRUE )
  importance( rf, type=1 )
}
fn.kill.wk()
fn.save.data("RFimp")
################################################################################################3

# Compute GBM Feature Importance for all 14 targets independently
# train <- as.data.frame(train)
# fn.register.wk(8)
# GBMimp <- foreach(fold=1:14, .combine=cbind, .inorder=TRUE, .packages = "gbm") %dopar% { # compute importance for all
#   tr <- train
#   tr$tgt <- TGT[,fold]
#   gbm1 <- gbm( as.formula("tgt ~ .") ,
#                distribution = "bernoulli",
#                data = tr,
#                n.trees = 200,
#                interaction.depth = 4,
#                n.minobsinnode = 5,
#                shrinkage = 0.05,
#                bag.fraction = 0.15,
#                keep.data = FALSE,
#                verbose = TRUE)
#   s <- summary(gbm1)
#   s$rel.inf
# }
# fn.kill.wk()
# fn.save.data("GBMimp")
################################################################################################3

fn.load.data("RFimp")
# fn.load.data("GBMimp")

plot( rowMeans(RFimp) )
feats <- which( rowMeans(RFimp) > 0  )
length(feats)
plot( rowMeans(RFimp)[feats] )

train <- train[,feats]
test <- test[,feats]
gc()


imp <- rowMeans(RFimp)[feats]

cv <- rep( 1:4 , length.out=nrow(train)   )

tic()
FEAT = 1
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_1 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_1")
toc()


tic()
FEAT = 2
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_2 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_2")
toc()

tic()
FEAT = 3
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_3 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_3")
toc()

tic()
FEAT = 4
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_4 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_4")
toc()

tic()
FEAT = 5
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_5 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_5")
toc()

tic()
FEAT = 6
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_6 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_6")
toc()

tic()
FEAT = 7
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_7 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_7")
toc()

tic()
FEAT = 8
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_8 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_8")
toc()

tic()
FEAT = 9
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_9 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_9")
toc()

tic()
FEAT = 10
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_10 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_10")
toc()

tic()
FEAT = 11
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_11 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_11")
toc()

tic()
FEAT = 12
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_12 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_12")
toc()

tic()
FEAT = 13
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_13 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_13")
toc()

tic()
FEAT = 14
rfA     <- rfCV( train, TGT[,FEAT], cv   , ite=480 )
rfAFULL <- rfALL( train, test, TGT[,FEAT], ite=480 )
fea <- which( imp < 0.90*max(imp) )
rfB     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfBFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
fea <- which( imp < 0.60*max(imp) )
rfC     <- rfCV( train[,fea], TGT[,FEAT], cv, ite=480 )
rfCFULL <- rfALL( train[,fea], test[,fea], TGT[,FEAT], ite=480 )
print(llfun(  TGT[,FEAT] ,   rfA    ))
print(llfun(  TGT[,FEAT] ,   rfB    ))
print(llfun(  TGT[,FEAT] ,   rfC    ))
print(llfun(  TGT[,FEAT] ,   (rfA + rfB + rfC)/3  ))
rf10_14 <- list( train = (rfA + rfB + rfC)/3 ,
                test  = (rfAFULL  + rfBFULL  + rfCFULL)/3  )
fn.save.data("rf10_14")
toc()

pred <- TGT
pred[ , 1 ] <- rf10_1$train
pred[ , 2 ] <- rf10_2$train
pred[ , 3 ] <- rf10_3$train
pred[ , 4 ] <- rf10_4$train
pred[ , 5 ] <- rf10_5$train
pred[ , 6 ] <- rf10_6$train
pred[ , 7 ] <- rf10_7$train
pred[ , 8 ] <- rf10_8$train
pred[ , 9 ] <- rf10_9$train
pred[ , 10 ] <- rf10_10$train
pred[ , 11 ] <- rf10_11$train
pred[ , 12 ] <- rf10_12$train
pred[ , 13 ] <- rf10_13$train
pred[ , 14 ] <- rf10_14$train
llfun( TGT , pred  )
# [1] 0.2531492
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
#0.2802321
# [1] 0.3858913
# [1] 0.4976762
# [1] 0.5085348
# [1] 0.03738223
# [1] 0.1602706
# [1] 0.06744872
# [1] 0.1737268
# [1] 0.4762454
# [1] 0.07111492
# [1] 0.3688261
# [1] 0.4679599
# [1] 0.1964832
# [1] 0.1676085
# [1] 0.3440809


sub <- fread('input/SubmissionFormat.csv')
sub <- data.frame(sub)
sub$service_a <- rf10_1$test
sub$service_b <- rf10_2$test
sub$service_c <- rf10_3$test
sub$service_d <- rf10_4$test
sub$service_e <- rf10_5$test
sub$service_f <- rf10_6$test
sub$service_g <- rf10_7$test
sub$service_h <- rf10_8$test
sub$service_i <- rf10_9$test
sub$service_j <- rf10_10$test
sub$service_k <- rf10_11$test
sub$service_l <- rf10_12$test
sub$service_m <- rf10_13$test
sub$service_n <- rf10_14$test

pred.rf.10 <- list(  train=pred , test=as.matrix(sub[ , 2:ncol(sub)  ]) )
fn.save.data("pred.rf.10")
str( pred.rf.10$train  )
str( pred.rf.10$test  )

# write.table( sub , "submission/sub.rf.10.csv", row.names=FALSE, quote=FALSE,sep="," )


