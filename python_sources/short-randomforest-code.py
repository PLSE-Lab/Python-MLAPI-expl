
# Any results you write to the current directory are saved as output.

# RandomForest

rm(list=ls())

##################  R Package checking #################
reqPackages = c("ranger", "ggplot2", "lattice", "ROCR","caret")
have = reqPackages %in% rownames(installed.packages())
if ( any(!have) ) { 
  cat("** Some required packages cannot be found. They are being installed now **\n")
  install.packages( reqPackages[!have] ) 
}

# load package
library(ranger)   # fast randomforest
library(ROCR)     # ROC curve
library(ggplot2)  # paper-ready figures


################## Load Data ###################################

# parameters
set.seed(123)     
iter = 100       #   # of iteration

# load rawdata
rawdata =read.csv('train.csv', header=T)    # Alcohol Problem
colnames(rawdata)[1] <- "DIAGNOSIS"
rawdata$DIAGNOSIS=as.factor(rawdata$DIAGNOSIS)


trainset = rawdata
testset = read.csv('test.csv',header = T)

  # build model
  rf <- ranger(DIAGNOSIS~. , data = trainset, importance = "impurity", write.forest = T, probability = F, classification = T)
  
  # test set
  rf.pr <- predict(rf, data = testset, type="response")
  rf.pr = rf.pr$predictions
  result_test <- data.frame('label'=c(1:28000), rf.pr)