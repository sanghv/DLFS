# Nguyen Ha Nam
# This function executes my Feature Selection method based on Deep learning
# It try until nomore feature can be eliminated and save the 
# learning accuracy into file 
#
# 2006.05.13 - save the cv accuracy into file
# 2006.05.19 - Execute the test part for each time of Fe elimination
# 2015.09.17 - Improved by sanghv
# 2015.10.26 - Repalce RF with Deep learning
## Load all packages first
suppressMessages(library(h2o))
suppressMessages(library(caret))
suppressMessages(library(mlbench))
suppressMessages(library(ggplot2))
suppressMessages(library(reshape2))
suppressMessages(library(deepnet))
source("code/start_timer.r")
source("code/stop_timer.r")
source("code/train_rbm.r")
source("lib/tictoc.r")
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
## Initialise H2O Connection
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#h2o.shutdown(conn = h2o.getConnection(), prompt = TRUE)
## Start a local H2O cluster directly from R
## For the first experiment, start the cluster with default 1GB RAM
#localH2O = h2o.init(ip = "localhost", port = 54321, nthreads = -1)
## Create an empty data frame for results

n_run <- 50
#n_epochs <- 50
res_tmp <- data.frame(Trial = 1:n_run, Training = NA, Test = NA, Duration = NA)
RFS<-function(){
  
  #=========================
  #Configuration
  #=========================
  path<-"data\\Cancer\\";
  inpF<-"ColonTumor.csv";
  header<-"ColonTumor_";
  inpF<-sprintf("%s%s", path,inpF);
  header<-sprintf("%s%s", path,header);
  
  FeImpType<-2;     # 1=mean decrease in accuracy 
  # 2=mean decrease in node impurity
  NormType<-0;
  
  MaxSim<-10;       # maximum number of simulation
  TrainingTime<-5; # folder number in n-folder crossvalidation method
  Step<-1;         # Speed up by set up some feature excepted in each iteration 
  n_epochs<-50;    # Random forest parameters
  LowerBound<-1;   # The alg will be stop when number of remain feature great or equal Lower bound
  
  pos_Status<-"positive";  # status for positive
  neg_Status<-"negative"; # status for negative
  flag_header<-TRUE;    # if TRUE: first line for colnames()
  # if FALSE: no column name 
 
  #outF<-sprintf("%sRFS_CV%d_TN%d_S%d_FImp%d_norm%d.csv", header, TrainingTime, 
  #						n_epochs, Step, FeImpType, NormType);
  
  InpData<-read.csv(inpF , strip.white = TRUE, header=flag_header);
  InpData<-InpData[,c(2:ncol(InpData))];
  # normalization
  #InpData<-norm(type=NormType, InpData);
  
  Neg_Data<-InpData[InpData[,ncol(InpData)]==neg_Status,];
  Pos_Data<-InpData[InpData[,ncol(InpData)]==pos_Status,];
  InpData[1:nrow(Neg_Data),]<-Neg_Data[1:nrow(Neg_Data),];
  InpData[(nrow(Neg_Data)+1):nrow(InpData),]<-Pos_Data[1:nrow(Pos_Data),];
  #colnames(InpData)<-c(1:(ncol(InpData)-1),"types");
  #colnames(InpData)<- c(paste(names(InpData[1:(ncol(InpData)-1)]),1:(ncol(InpData)-1),sep = "_"),"types")
  colnames(InpData)<- c(paste("",1:(ncol(InpData)-1),sep = ""),"types")
  #InpData<-InpData[, c(1:(ncol(InpData)-1),"types")];
  
  CanNum<-nrow(InpData[InpData[,ncol(InpData)]==neg_Status,]);
  NorNum<-nrow(InpData[InpData[,ncol(InpData)]==pos_Status,]);
  
  CanTrainNum<-round(CanNum/2);
  NorTrainNum<-round(NorNum/2);
  
  CanTrainStep<-round(CanTrainNum/TrainingTime);
  NorTrainStep<-round(NorTrainNum/TrainingTime);
  outFcv<-sprintf("%sRFS_CV%d_TN%d_Ftype%d_Learn.csv", header, TrainingTime,n_epochs,FeImpType);
  outFcvset<-matrix(data=NA, ncol=MaxSim, nrow=(ncol(InpData)-2));
  outFpred<-sprintf("%sRFS_CV%d_TN%d_Ftype%d_pred.csv", header, TrainingTime,n_epochs,FeImpType);
  outFpredset<-matrix(data=NA, ncol=MaxSim, nrow=(ncol(InpData)-2));
  
  #=========================
  #library("randomForest");
  for (i in 1:MaxSim) {
    #browser()
    localH2O <<- h2o.init(ip = "localhost", port = 54321, nthreads = -1)
    #Randomize choose sample for training and testing set
    ranCan<-sample(1:CanNum);
    ranNor<-sample((CanNum+1):nrow(InpData));
    
    ind<-NA;
    stCanPos<-1;   
    stNorPos<-1;   
    for(tt in 1:TrainingTime) {
      ind[ranCan[stCanPos:(tt*CanTrainStep)]]= tt;
      ind[ranNor[stNorPos:(tt*NorTrainStep)]]= tt;
      stCanPos<-tt*CanTrainStep+1;   
      stNorPos<-tt*NorTrainStep+1;   
    }
    ind[ranCan[stCanPos:CanNum]]= TrainingTime+1;
    ind[ranNor[stNorPos:NorNum]]= TrainingTime+1;
  #  InpData<-as.h2o(localH2O,InpData)
    resdat<-Processing(InpData, n_epochs, Step, ind, 
                       FeImpType, TrainingTime, outF, LowerBound, rtime);  
    outF1<-sprintf("%sbackward_CV%d_TN%d_Ftype%d_%d.csv", header, TrainingTime, 
                   n_epochs,FeImpType, i);
    write.csv(resdat, outF1, row.names = T);
    outFcvset[,i]<-resdat[,3];
    outFpredset[,i]<-resdat[,4];
    h2o.shutdown(conn = h2o.getConnection(), prompt = F)
  }
  
  write.csv(outFcvset, outFcv, row.names = T);
  write.csv(outFpredset, outFpred, row.names = T);
  #==========================================================
  outFAveM<-sprintf("%sRFS_CV%d_TN%d_Ftype%d_Ave.csv", header, TrainingTime,n_epochs,FeImpType);
  AveM<-matrix(data=NA, ncol=(ncol(InpData)-2), nrow=2);
  for(i in 1:(ncol(InpData)-2) ) {
    AveM[1, i]<-mean(outFcvset[i, ]);
    AveM[2, i]<-mean(outFpredset[i, ]);
  }
  #browser();
  row.names(AveM)<-c("Learn Acc.", "Prediction Acc.");
  write.csv(AveM, outFAveM);
}

###########################################################################
# Execute Deep learning 
DL_exec<-function(train_hex, test_hex, n_epochs, FeImpType) {
  tic()
  ## Train the model
  #browser()
#   train_hex <- as.h2o(TrainData,localH2O, destination_frame = "train_hex")
#   test_hex <- as.h2o(TestData,localH2O,destination_frame = "test_hex")
#   y_train <- as.factor(train_hex$types)
#   y_test <- as.factor(test_hex$types)
#   
  nc<-ncol(train_hex)
 
  model <- h2o.deeplearning(x = 1:(nc-1),  # column numbers for predictors
                            y = nc,   # column number for label
                            training_frame =  train_hex,
                            activation = "Tanh",
                            balance_classes = TRUE,
                            hidden = c(50,50,50),  ## three hidden layers
                            variable_importances = T,
                            epochs = n_epochs)
  pred<-h2o.predict(model, test_hex)
# browser()
  out<-list();
  browser()
  ## Store Z-Scores
  vi<-model@model$variable_importances#[,1:2]
  ## Sort
 # order_v <- order(vi$variable, decreasing = T)
#  m<-vi
  # m is your matrix
  vi <- vi[order(as.integer(vi$variable)),]
 # vi <- vi[order_v, ]
  out$FeImp<-as.numeric(vi[,2])
# browser()
  #confusion from h2o
  confusion<-h2o.confusionMatrix(model)
  out$LearnAcc <- (confusion[1,1]+confusion[2,2])/
    (confusion[1,1]+confusion[2,2]+confusion[2,1]+confusion[1,2]);
  ##########3
  out$LearnAUC<- model@model$training_metrics@metrics$AUC
  # compare metrics
  perf<-h2o.performance(model,test_hex)
  confusion<-h2o.confusionMatrix(perf)
  out$TestAcc <- (confusion[1,1]+confusion[2,2])/
    (confusion[1,1]+confusion[2,2]+confusion[2,1]+confusion[1,2]);
    # browser()
#   out$res <- confusionMatrix(pr,te)$overall[1]#TestData[, ncol(TestData)], 
#   out$TestAcc <-out$res
  cat("\n DL:")
  toc()
  return(out);
}
##########################################################################
CV_DL_exec<-function(inpData, n_epochs, ind, CrossValid, FeImpType) {
  tic()
  SumLeAcc<-0;
  SumTeAcc<-0;
  for(cv in 1:CrossValid){
    TestData<-inpData[ind==cv,];
    TrainData<-inpData[ind!=cv,];
  # browser()
    train_hex <- as.h2o(TrainData,localH2O, destination_frame = "train_hex")
    test_hex <- as.h2o(TestData,localH2O,destination_frame = "test_hex")
    out<-DL_exec(train_hex, test_hex, n_epochs, FeImpType);
    #out<-DL_exec(TrainData, TestData, n_epochs, FeImpType);
    SumLeAcc<-SumLeAcc+out$LearnAcc;
    SumTeAcc<-SumTeAcc+out$TestAcc;
    #Ranking fomular of Feature Importance 
   # browser();
#     weight<-abs(out$LearnAcc-out$TestAcc)/(out$Learn+out$TestAcc);
#     if (weight==0) 
#       weight<-1
#     else
#       weight<-weight;
#modify by sanghv 
    rt<-out$LearnAcc-out$TestAcc
    if(rt==0)
      weight<-1
    else
      weight<-abs(out$LearnAcc-out$TestAcc)/(out$LearnAcc+out$TestAcc);
    if(cv==1)
      
      FeImp<-out$FeImp*weight
    else
      FeImp<-FeImp + out$FeImp*weight;
  }
  
  res<-list();
  res$FeImp<-FeImp;
  res$LearnAcc<-SumLeAcc/CrossValid;
  res$TestAcc<-SumTeAcc/CrossValid;
  cat("\n CV:")
  toc()
  return(res);
}

###############################################################################
new_FS_build<-function(FeSelected, BestFeS, Step, FeImpType) {
  browser()
  out<-list();
  tmp_FeSelected<-FeSelected; 
  #improved bay sanghv: remove feature with vip<0
  mat.imp <- as.matrix(BestFeS)
  mat.imp <- as.vector(mat.imp)#convert to
  filt.var <- names(mat.imp[which(mat.imp[,1] > 0),])
  tmp_FeSelected<-as.integer(substring(filt.var, 2))
  ###########
  for(step in 1:Step) {
    if(length(tmp_FeSelected)>1) {
      minFeSpos<-which.min(BestFeS);
      minvalue<-FeSelected[minFeSpos];
      tmp_FeSelected<-tmp_FeSelected[tmp_FeSelected!=minvalue];
      BestFeS[minFeSpos]<- max(BestFeS)+1;
      out$minFeSpos<-minFeSpos;
    }
  }
  #browser();
  
  out$FeSelected<-tmp_FeSelected;
  return(out);
}

###############################################################################

Processing<-function(inpData, n_epochs, Step, ind, 
                     FeImpType,TrainTime, outF, LowerBound, rtime) {

  #cat("Index, Feature Imp. Ind, CV Acc., Pred. Accuracy\n", file=outF, append=FALSE);
  resmatrix<-list(); i<-0;
  resmatrix$Index<-NA; resmatrix$FeInd<-NA; resmatrix$CV<-NA; resmatrix$Pred<-NA;
  
  FeSelected<-1:(ncol(inpData)-1);#create feature selection list
  #browser()
  cv_out<-CV_DL_exec(inpData, 
                     n_epochs, ind, TrainTime, FeImpType)
  train_hex <- as.h2o(inpData[ind<=TrainTime,],localH2O, destination_frame = "train_hex")
  test_hex <- as.h2o(inpData[ind==(TrainTime+1),],localH2O,destination_frame = "test_hex")
  #out<-DL_exec(inpData[ind<=TrainTime,], inpData[ind==(TrainTime+1),], n_epochs, FeImpType)
  out<-DL_exec(train_hex, test_hex , n_epochs, FeImpType)
  
  BestFeS<-cv_out$FeImp;
  BestAcc<-cv_out$TestAcc;
  tmp_Data<-inpData;
  keepTrack<-0;
  
  #tmp_FeSelected<-FeSelected;
  #LowerBound<-LowerBound+FeImpType;
  iteration<-0;
  while(length(FeSelected) >LowerBound) {
    cat("FS:", (ncol(tmp_Data)-1),"Track: ",keepTrack, 
                ", (Te:", cv_out$TestAcc, 
                ", Best: ", BestAcc, ")\n");
    cat(".");
    iteration<-iteration+1; 
     cat("Iteration = ", iteration,"->");
    
    #find minimum possition in FeImp vector
    out_FS<-new_FS_build(FeSelected, BestFeS, Step, FeImpType);
    tmp_FeSelected<-out_FS$FeSelected;
    minFeSpos<-out_FS$minFeSpos;
    print(tmp_FeSelected);
    #browser()
    tmp_Data<-inpData[,c(tmp_FeSelected,ncol(inpData))];
    if(is.null(tmp_Data)) break;
    cv_out<-CV_DL_exec(tmp_Data, 
                       n_epochs, ind, TrainTime, FeImpType);
    
    if(cv_out$TestAcc>=BestAcc) {
      cat("-");
      i<-i+1;
      resmatrix$Index[i]<-minFeSpos; 
      resmatrix$FeInd[i]<-FeSelected[minFeSpos];
      resmatrix$CV[i]<-cv_out$TestAcc;
      resmatrix$Pred[i]<-out$TestAcc;
      #cat(minFeSpos, FeSelected[minFeSpos], cv_out$TestAcc, out$TestAcc, "\n", file=outF, sep=",", append=TRUE);
      
      cat((length(FeSelected)-1), " -- out:", FeSelected[minFeSpos], ", ", cv_out$TestAcc, ", ", out$TestAcc, "\n");
      
      BestFeS<-cv_out$FeImp;
      BestAcc<-cv_out$TestAcc;
      FeSelected<-tmp_FeSelected;
      keepTrack<-0; 
      if(length(FeSelected)<10) cat(FeSelected,"\n");
      {
        train_hex <- as.h2o(tmp_Data[ind<=TrainTime,],localH2O, destination_frame = "train_hex")
        test_hex <- as.h2o(tmp_Data[ind==(TrainTime+1),],localH2O,destination_frame = "test_hex")
        out<-DL_exec(train_hex, test_hex, n_epochs, FeImpType)
     # out<-DL_exec(tmp_Data[ind<=TrainTime,], tmp_Data[ind==(TrainTime+1),], n_epochs, FeImpType)
      }
    }
    else if (keepTrack<=(ncol(tmp_Data)-LowerBound)){
      if(keepTrack==0) {
        minFeSpos1<-minFeSpos;
        BestFeS1<-cv_out$FeImp;
        BestAcc1<-cv_out$TestAcc;
        FeSelected1<-tmp_FeSelected;
        Pred<-out$TestAcc;
      }
      keepTrack<-keepTrack+1;
      BestFeS[minFeSpos]<-max(BestFeS)+1;
    }
    
    #stop if can not find any feature to increase the accuracy
    if(keepTrack>(ncol(tmp_Data)-LowerBound)) {
      cat("#");
      cat((length(FeSelected)-1), "+ out:", FeSelected[minFeSpos1], ", ", BestAcc1,  ", ", Pred, "\n");
      i<-i+1;
      resmatrix$Index[i]<-minFeSpos1; 
      resmatrix$FeInd[i]<-FeSelected[minFeSpos1];
      resmatrix$CV[i]<-BestAcc1;
      resmatrix$Pred[i]<-Pred;
      #cat(minFeSpos1, FeSelected[minFeSpos1], BestAcc1, ", ", Pred, "\n", file=outF, sep=",", append=TRUE);
      
      BestFeS<-BestFeS1;
      BestAcc<-BestAcc1;
      FeSelected<-FeSelected1;
      keepTrack<-0; 
      #out<-DL_exec(tmp_Data[ind<=TrainTime,], tmp_Data[ind==(TrainTime+1),], n_epochs, FeImpType)
      #if(length(FeSelected)<10) cat(FeSelected,"\n");
      #break;
    }
    
  }#End of while-do until no feature in FeSelected set
  return(data.frame(resmatrix));
}


# You can add new normalisation method here
norm<-function(type=1, InpData) {
  if(type==1){
    # min-max normalization   
    for (fInd in 1: (ncol(InpData)-1)) {
      InpData[,fInd]=(InpData[,fInd]- min(InpData[,fInd]))/(max(InpData[,fInd])-min(InpData[,fInd]));
    }
  }else if(type==2) {
    # Normalization based on Gose book, page 171
    #inp<-sample(10,12,replace=TRUE);
    #dim(inp)<-c(3,4);
    #InpData<-inp;
    for (fInd in 1: (ncol(InpData)-1)) {
      Med<-median(InpData[,fInd]);
      MAD<-sum(abs(InpData[, fInd]-Med))/nrow(InpData);
      InpData[,fInd]<-(InpData[,fInd]-Med)/MAD;
    }
  }
  return(InpData);
}