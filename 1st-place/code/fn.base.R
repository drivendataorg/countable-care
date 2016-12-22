require("data.table")
require("compiler")
library(randomForest)
library(miscTools)
library("Metrics")

require("compiler")
enableJIT(3) 
setCompilerOptions(suppressUndefined = T)
options(stringsAsFactors = FALSE)
path.wd <- getwd()

all.noexport <- character(0)


#############################################################
# tic toc
#############################################################
tic <- function(gcFirst = TRUE, type=c("elapsed", "user.self", "sys.self")) {
  type <- match.arg(type)
  assign(".type", type, envir=baseenv())
  if(gcFirst) gc(FALSE)
  tic <- proc.time()[type]         
  assign(".tic", tic, envir=baseenv())
  invisible(tic)
}

toc <- function() {
  type <- get(".type", envir=baseenv())
  toc <- proc.time()[type]
  tic <- get(".tic", envir=baseenv())
  print(toc - tic)
  invisible(toc)
}

##############################################################
## Registers parallel workers
##############################################################
fn.register.wk <- function(n.proc = NULL) {
  if (file.exists("data/cluster.csv")) {
    cluster.conf <- read.csv("data/cluster.csv", 
                             stringsAsFactors = F,
                             comment.char = "#")
    n.proc <- NULL
    for (i in 1:nrow(cluster.conf)) {
      n.proc <- c(n.proc, 
                  rep(cluster.conf$host[i], 
                      cluster.conf$cores[i]))
    }
  }
  if (is.null(n.proc)) {
    n.proc = as.integer(Sys.getenv("NUMBER_OF_PROCESSORS"))
    if (is.na(n.proc)) {
      library(parallel)
      n.proc <-detectCores()
    }
  }
  workers <- mget(".pworkers", envir=baseenv(), ifnotfound=list(NULL));
  if (!exists(".pworkers", envir=baseenv()) || length(workers$.pworkers) == 0) {
    
    library(doSNOW)
    library(foreach)
    workers<-suppressWarnings(makeSOCKcluster(n.proc));
    suppressWarnings(registerDoSNOW(workers))
    clusterSetupRNG(workers, seed=5478557)
    assign(".pworkers", workers, envir=baseenv());
    
    #     tic()
    #     cat("Workers start time: ", format(Sys.time(), 
    #                                        format = "%Y-%m-%d %H:%M:%S"), "\n")
  }
  invisible(workers);
}

##############################################################
## Kill parallel workers
##############################################################
fn.kill.wk <- function() {
  library("doSNOW")
  library("foreach")
  workers <- mget(".pworkers", envir=baseenv(), ifnotfound=list(NULL));
  if (exists(".pworkers", envir=baseenv()) && length(workers$.pworkers) != 0) {
    stopCluster(workers$.pworkers);
    assign(".pworkers", NULL, envir=baseenv());
    #     cat("Workers finish time: ", format(Sys.time(), 
    #                                         format = "%Y-%m-%d %H:%M:%S"), "\n")
    #     toc()
  }
  invisible(workers);
}

##############################################################
## init worker setting work dir and doing path redirect
##############################################################
fn.init.worker <- function(log = NULL, add.date = FALSE) {
  #log<- paste("gbm17_v2_k",fold,sep="")
  #add.date = FALSE
  setwd(path.wd)
  
  if (!is.null(log)) {
    date.str <- format(Sys.time(), format = "%Y-%m-%d_%H-%M-%S")
    
    if (add.date) {
      output.file <- fn.log.file(paste(log, "_",date.str,
                                       ".log", sep=""))
    } else {
      output.file <- fn.log.file(paste(log,".log", sep=""))
    }
    output.file <- file(output.file, open = "wt")
    sink(output.file)
    sink(output.file, type = "message")
    
    #    cat("Start:", date.str, "\n")
  }
  tic()
}

##############################################################
## clean worker resources
##############################################################
fn.clean.worker <- function() {
  gc()
  
  try(toc(), silent=T)
  suppressWarnings(sink())
  suppressWarnings(sink(type = "message"))
}

#############################################################
# log file path
#############################################################
fn.log.file <- function(name) {
  paste(path.wd, "log", name, sep="/")
}


#############################################################
# data file path
#############################################################
fn.data.file <- function(name) {
  paste(path.wd, "data", name, sep="/")
}
#############################################################
# save data file
#############################################################
fn.save.data <- function(dt.name, envir = parent.frame()) {
  save(list = dt.name, 
       file = fn.data.file(paste0(dt.name, ".RData")), envir = envir)
}
#############################################################
# load saved file
#############################################################
fn.load.data <- function(dt.name, envir = parent.frame()) {
  load(fn.data.file(paste0(dt.name, ".RData")), envir = envir)
}



rmse <- function(a,b){
  r<- sqrt(mean((a-b)^2))
  r
}
sna <- function(x){
  r<-sum(is.na(x))
  r
}
Mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

colRank <- function(X) apply(X, 2, rank)
colMedian <- function(X) apply(X, 2, median)
colMax <- function(X) apply(X, 2, max)
colMin <- function(X) apply(X, 2, min)
colSd <- function(X) apply(X, 2, sd)
mae <- function(c1,c2) {
  c1 <- as.numeric(c1)
  c2 <- as.numeric(c2)
  score <- mean( abs(c1-c2) )
  score
}
rep.row<-function(x,n){
  matrix(rep(x,each=n),nrow=n)
}
rep.col<-function(x,n){
  matrix(rep(x,each=n), ncol=n, byrow=TRUE)
}
gc()
rowMax <- function(X) apply(X, 1, max)
rowMin <- function(X) apply(X, 1, min)
rowMean <- function(X) apply(X, 1, mean,na.rm=TRUE)
rowSd <- function(X) apply(X, 1, sd)
rowMode <- function(X) apply(X, 1, Mode)
colAUC <- function(X,y) apply(X, 2, getROC_AUC, y)







getROC_AUC = function(probs, true_Y){
  probsSort = sort(probs, decreasing = TRUE, index.return = TRUE)
  val = unlist(probsSort$x)
  idx = unlist(probsSort$ix)  
  
  roc_y = true_Y[idx];
  stack_x = cumsum(roc_y == 0)/sum(roc_y == 0)
  stack_y = cumsum(roc_y == 1)/sum(roc_y == 1)    
  
  auc = sum((stack_x[2:length(roc_y)]-stack_x[1:length(roc_y)-1])*stack_y[2:length(roc_y)])
  return(auc)
}



gbmCV <- function( tr , ts , tgt , cv , ite=100 ,shri=0.1, depth=3, verbose=FALSE ){
  #   tr <- train[,1:10]
  #   ts <- train[,1:10]
  #   tgt <- target$Ca
  #   nfolds=5
  # ite=100
  nfolds = max(cv)
  pred.train <- rep( 0 , nrow(tr))
  pred.test  <- rep( 0 , nrow(ts))
  
  for( fold in 1:nfolds  ){
    
    px <- which( cv!=fold   )
    py <- which( cv==fold   )
    
    g <- gbm.fit(x=tr[px,], y=tgt[px],
                 distribution = "bernoulli",
                 n.trees = ite ,
                 interaction.depth = depth,
                 n.minobsinnode = 10,
                 shrinkage = shri,
                 bag.fraction = 0.50,
                 keep.data = TRUE,
                 verbose = TRUE)
    pred.train[py] <- predict(g , tr[py,],n.trees=ite ,type="response")
    if( verbose==TRUE ){
      print(paste( fold , llfun(tgt[py],pred.train[py]) , llfun(tgt[which(cv<=fold)],pred.train[which(cv<=fold)]) ) )
    }
    
    pred.test <- pred.test + predict(g , ts,n.trees=ite ,type="response")/nfolds
  }
  list( train=pred.train , test=pred.test , summary=summary(g,plot=FALSE) )
}


rfCV <- function( tr, tgt, cv, ite=100 ){
  
  tgt <- factor(tgt)
  
  fn.register.wk(4)
  res <- foreach(fold=unique(cv), .combine=rbind, .inorder=TRUE, .packages = "randomForest") %dopar% {
    rf <- randomForest( x=tr[which( cv!=fold  ),] , y=tgt[which( cv!=fold  )] , ntree=ite )
    pred <- predict(rf ,  tr[which( cv==fold  ),], type="prob")[,2]
    cbind( which( cv==fold  ) , pred )    
  }
  fn.kill.wk()
  res <- res[ order(res[,1])  ,]
#   rf <- randomForest( x=tr , y=tgt , ntree=ite )
#   pred <- predict(rf ,  ts, type="prob")[,2]
#   
#   list( train=as.numeric( res[ , 2] ) , test=pred )
  as.numeric( res[ , 2] )
}

rfALL <- function( tr, ts, tgt, ite=16 ){
  
  tgt <- factor(tgt)
  
  fn.register.wk(8)
  res <- foreach(fold=1:8, .combine=cbind, .inorder=FALSE, .packages = "randomForest") %dopar% {
    rf <- randomForest( x=tr , y=tgt , ntree=round(ite/8) ) 
    pred <- predict(rf , ts, type="prob")[,2]
    as.numeric(pred)
  }
  fn.kill.wk()
  
  as.numeric( rowMeans(res) )
}



rfRegCV <- function( tr, ts, tgt, cv, ite=100 ){
  
  pred <- rep( 0 , nrow(tr))
  for(fold in unique(cv)) {
    px <- which( cv!=fold   )
    py <- which( cv==fold   )
    rf <- randomForest( x=tr[px,] , y=tgt[px] , ntree=ite, corr.bias=FALSE )
    pred[py] <- as.numeric( predict(rf ,  tr[py,]) )
  }
  
  rf <- randomForest( x=tr , y=tgt , ntree=ite, corr.bias=FALSE )
  predtest <- as.numeric(predict(rf ,  ts))
  
  list( train=pred , test=predtest )
}





svmCV <- function( tr , ts , tgt , cv , cost=10, verbose=FALSE ){
  #   tr <- train[,1:10]
  #   ts <- train[,1:10]
  #   tgt <- target$Ca
  #   nfolds=5
  # ite=100
  nfolds = max(cv)
  pred.train <- rep( 0 , nrow(tr))
  pred.test  <- rep( 0 , nrow(ts))
  
  for( fold in 1:nfolds  ){
    
    px <- which( cv!=fold   )
    py <- which( cv==fold   )
    
    g <- svm(x=tr[px,], y=tgt[px], cost = cost)
    pred.train[py] <- predict(g , tr[py,],type="response")
    
    
    if( verbose==TRUE ){
      print(paste( fold , llfun(pred.train[py],tgt[py]) , llfun(pred.train[which(cv<=fold)], tgt[which(cv<=fold)]) ) )
    }
    
    pred.test <- pred.test + predict(g , ts,n.trees=ite ,type="response")
  }
  list( train=pred.train , test=pred.test/nfolds) 
}




glmFACTOR <- function( tr, ts, target , cv  ){
  #   tr <- TR[px,]
  #   ts <- TR[py,]
  #   target <- TGT[,6]
  #   fold=1
  nfolds = max(cv)
  pred.train <- rep( 0 , nrow(tr))
  pred.test  <- rep( 0 , nrow(ts))
  tr$tgt <- target
  ts$tgt <- 0
  for( fold in 1:nfolds  ){
      g <- glm( tgt ~ . , tr[which(cv!=fold),] , family="binomial" )
      pred.train[which(cv==fold)] <- predict.glm(g , tr[which(cv==fold),] ,type="response" )
  }

  g <- glm( tgt ~ . , tr , family="binomial"  )
  pred.test <- predict.glm(g , ts,type="response" )
  list( train=as.numeric( pred.train ), test=as.numeric( pred.test ) )
}

lmFACTOR <- function( tr, ts, target , cv  ){
  #   tr <- tr1
  #   target <- TGT[,1]
  #   fold=1
  nfolds = max(cv)
  pred.train <- rep( 0 , nrow(tr))
  tr$tgt <-  target 
  for( fold in 1:nfolds  ){
    px <- which( cv!=fold   )
    py <- which( cv==fold   )
    g <- lm( tgt ~ . , tr[px,], model=FALSE   )
    pred.train[py] <- predict(g , tr[py,] ,type="response" )
  }
  g <- glm( tgt ~ . , tr , family="binomial"  )
  pred.test <- predict(g , ts,type="response" )
  
  list( train=as.numeric( pred.train ), test=as.numeric( pred.test ) )
}




lmCV <- function( tr , ts , tgt , cv , verbose=FALSE ){
  #   tr <- train[,1:10]
  #   ts <- train[,1:10]
  #   tgt <- target$Ca
  #   nfolds=5
  # ite=100
  nfolds = max(cv)
  pred.train <- rep( 0 , nrow(tr))
  pred.test  <- rep( 0 , nrow(ts))
  tr$tgt <- tgt
  
  for( fold in 1:nfolds  ){
    
    px <- which( cv!=fold   )
    py <- which( cv==fold   )
    
    #    g <- glm( tgt ~ . , tr[px,],family=gaussian()  )
    g <- lm( tgt ~ . , tr[px,]  )
    
    pred.train[py] <- predict(g , tr[py,] ,type="response" )
    
    if( verbose==TRUE ){
      print(paste( fold , rmse(pred.train[py],tgt[py]) , rmse(pred.train[which(cv<=fold)],tgt[which(cv<=fold)]) ) )
    }
    pred.test <- pred.test + predict(g , ts ,type="response")/nfolds
  }
  list( train=pred.train , test=pred.test )
}




xgbCV <- function( tr , ts , tgt, MTGT, cv , ite=100 ,shri=0.1, depth=3,subsample=0.5,colsample=1.0, verbose=FALSE, ncore=4 ){
  #   tr <- train
  #   ts <- test
  #   tgt <- TGTi
  # MTGT = TGT
  #   nfolds=max(cv)
  # ite=10
  # shri =0.1
  # depth=3
  # subsample = 0.5
  # colsample = 0.5
  tr <- as.matrix(tr)
  ts <- as.matrix(ts)
  nfolds = max(cv)
  pred.train <- matrix( 0 ,nrow=nrow(tr), ncol=length(unique(tgt)) )
  pred.test  <- matrix( 0 ,nrow=nrow(ts), ncol=length(unique(tgt)) )
  
  xgmatTSS <- xgb.DMatrix( ts, missing = -999.0)
  fold=1
  for( fold in unique(cv) ){
    px <- which( cv!=fold   )
    py <- which( cv==fold   )
    xgmat   <- xgb.DMatrix( tr[px,], label = tgt[px], missing = -999.0)
    xgmatTS <- xgb.DMatrix( tr[py,], label = tgt[py], missing = -999.0)
    param <- list("objective" = "multi:softprob",
                  "num_class" = length(unique(tgt)) ,
                  "bst:eta" = shri,
                  "bst:max_depth" = depth ,
                  "subsample" = subsample,
                  "colsample_bytree" = colsample ,
                  "silent" = 1,
                  "eval_metric" = "mlogloss",
                  "nthread" = ncore)
    watchlist <- list("test"=xgmatTS)
    bst = xgb.train(param, xgmat, ite , watchlist)
    y <- predict(bst, xgmatTS, ntreelimit=ite)
    y <- matrix( y , nrow=length(py) , ncol=length(unique(tgt))  , byrow=TRUE )
    pred.train[py, ] <- y
    gc()
    if( verbose==TRUE ){
          print(paste( fold , llfun(MTGT[py,],pred.train[py,]) , llfun(MTGT[which(cv<=fold),],pred.train[which(cv<=fold),]) ) )
    }
    y <- predict(bst, xgmatTSS, ntreelimit=ite)
    y <- matrix( y , nrow=nrow(ts) , ncol=ncol(MTGT)  , byrow=TRUE )
    pred.test <- pred.test + y
  }

  list( train=pred.train , test=pred.test/nfolds )
}



xgbCV2 <- function( tr , ts , tgt,  cv, ite=100 ,shri=0.1, depth=3,subsample=0.5,colsample=1.0, verbose=FALSE, trainFULL=TRUE, ncores=4 ){
  #   tr <- train[,1:550]
  #   ts <- test[,1:550]
  #   tgt <- TGT[,FEAT]
  #   nfolds=4
  # ite=200
  # shri =0.1
  # depth=4
  # subsample = 0.25
  # colsample = 0.25
  tr <- as.matrix(tr)
  ts <- as.matrix(ts)
  nfolds = max(cv)
  pred.train <- rep( 0 ,nrow=nrow(tr))
  pred.test  <- rep( 0 ,nrow=nrow(ts))
  xgmatTSS <- xgb.DMatrix( ts, missing = -999.0)
  fold=1
  for( fold in unique(cv) ){
    px <- which( cv!=fold   )
    py <- which( cv==fold   )
    xgmat   <- xgb.DMatrix( tr[px,], label = tgt[px], missing = -999.0)
    xgmatTS <- xgb.DMatrix( tr[py,], label = tgt[py], missing = -999.0)
    param <- list("objective" = "binary:logistic",
                  "bst:eta" = shri,
                  "bst:max_depth" = depth ,
                  "subsample" = subsample,
                  "colsample_bytree" = colsample ,
                  "eval_metric" = "logloss", "nthread" = ncores, "silent" = 1)
    watchlist <- list("test"=xgmatTS)
    if( verbose==TRUE ){
      bst = xgb.train(param, xgmat, ite , watchlist)
    }else{
      bst = xgb.train(param, xgmat, ite) 
    }
    xgb.save( bst, paste0( "train_fold",fold,'.xgb') )
    pred.train[py] <- predict(bst, xgmatTS, ntreelimit=ite)
    if( verbose==TRUE ){
      print(paste( fold , llfun(tgt[py],pred.train[py]) , llfun(tgt[which(cv<=fold)], pred.train[which(cv<=fold)]) ) )
    }
#     pred.test <- pred.test + predict(bst, xgmatTSS, ntreelimit=ite)/nfolds
    gc()
  }
  
  best = llfun(tgt,pred.train)
  best.ite=ite
  if( verbose==TRUE ){
    print('Searching best tree number...')
#     print(paste(best.ite,best))
  }
  for( it in seq(round(2*ite/3),ite,1+round(0.02*ite))  ){
    pred <- rep( 0 ,nrow=nrow(tr))
    for( fold in unique(cv) ){
      py <- which( cv==fold   )
      xgmatTS <- xgb.DMatrix( tr[py,], missing = -999.0)
      bst <- xgb.load( paste0( "train_fold",fold,'.xgb') )
      pred[py] <- predict(bst, xgmatTS, ntreelimit=it)
    }
    sc <- llfun(  tgt , pred )
#     print( paste( it , sc )  )
    if( sc < best  ){
      best.ite = it
      best = sc
    }
  }
#   if( verbose==TRUE ){
    print(  paste( best.ite, best ) )
#   }
  
  for( fold in unique(cv) ){#Predict with the best number of iterations
    py <- which( cv==fold   )
    xgmatTS <- xgb.DMatrix( tr[py,], missing = -999.0)
    bst <- xgb.load( paste0( "train_fold",fold,'.xgb') )
    pred.train[py] <- predict(bst, xgmatTS, ntreelimit=best.ite)
    pred.test <- pred.test + predict(bst, xgmatTSS, ntreelimit=best.ite)/nfolds
  }
  
  pred.test.full = numeric()
  if( trainFULL==TRUE ){
    xgmat   <- xgb.DMatrix( tr, label = tgt, missing = -999.0)
    bst = xgb.train(param, xgmat, ite )
    pred.test.full <- predict(bst, xgmatTSS, ntreelimit = best.ite)
  }
  list( train=pred.train , test=pred.test , testFULL=pred.test.full )
}







MSMSE <- function(pred, actual){
  p <- rmse( pred[,1] , actual[,1])
  p <- p+rmse( pred[,2] , actual[,2])
  p <- p+rmse( pred[,3] , actual[,3])
  p <- p+rmse( pred[,4] , actual[,4])
  p <- p+rmse( pred[,5] , actual[,5])
  print(rmse( pred[,1] , actual[,1]))
  print(rmse( pred[,2] , actual[,2]))
  print(rmse( pred[,3] , actual[,3]))
  print(rmse( pred[,4] , actual[,4]))
  print(rmse( pred[,5] , actual[,5]))
  
  p/5  
}



llfun <- function(actual, prediction) {
  epsilon <- 1e-16
  yhat <- pmin(pmax(prediction, epsilon), 1-epsilon)
  logloss <- -mean(actual*log(yhat)
                   + (1-actual)*log(1 - yhat))
  return(logloss)
}

LogLoss <- function( actual , prediction   ){
  s <- 0
  for( i in 1:14  ){
    s <- s + llfun( actual[,i] , prediction[,i] )
    print(paste( i, llfun( actual[,i] , prediction[,i] )) )
  }
  s/14
}


nllmc <- function( apriori , predicted ){
  s <- rowSums(predicted)
#   s[s< 1e-15 ] <- 1e-15
#   s[s>(1-1e-15)  ] <- (1-1e-15)
  s <- matrix( rep(s,ncol(apriori)) , nrow=length(s) , ncol=ncol(apriori) , byrow=FALSE  )
  predicted <- predicted / s
  predicted[predicted< 1e-15] <- 1e-15
  predicted[predicted> (1-1e-15)] <- (1-1e-15)
  s <- rowSums(predicted)
  -sum( apriori * log( predicted )  )/nrow(predicted)
}
llmc <- function( apriori , predicted ){
  predicted[predicted< 1e-15] <- 1e-15
  predicted[predicted> (1-1e-15)] <- (1-1e-15)
  s <- rowSums(predicted)
  -sum( apriori * log( predicted )  )/nrow(predicted)
}






