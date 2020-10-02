# This script uses the Naive Bayes classifier based on the data,
# saves a sample submission, also uses klaR package for plots
# 

library(ggplot2)
library(C50)
library(e1071)
library(klaR)

set.seed(1)
train <- read.csv("../input/train.csv", stringsAsFactors=FALSE)
test  <- read.csv("../input/test.csv",  stringsAsFactors=FALSE)

extractFeatures <- function(data) {
  features <- c("Pclass",
                "Age",
                "Sex",
                "Parch",
                "SibSp",
                "Fare",
                "Embarked")
  fea <- data[,features]
  
  fea$Age[is.na(fea$Age)] <- -1
  fea$Fare[is.na(fea$Fare)] <- median(fea$Fare, na.rm=TRUE)
  fea$Embarked[fea$Embarked==""] = "S"
  fea$Sex      <- as.factor(fea$Sex)
  fea$Embarked <- as.factor(fea$Embarked)
  return(fea)
}

extract_train <-extractFeatures(train)
extract_test <-extractFeatures(test)

nb <- naiveBayes(extract_train,as.factor(train$Survived))
submission <- data.frame(PassengerId = test$PassengerId)
submission$Survived<- predict(nb,as.data.frame(extract_test))
write.csv(submission, file = "1_naive_bayes_r_submission.csv", row.names=FALSE)


extract_train <- extract_train[-7]

nbS <- NaiveBayes(Sex ~ . , data = extract_train)
pdf("SampleGraph_traindata.pdf",width=7,height=5)
x=rnorm(100)
y=rnorm(100,5,1)
p<- plot(nbS)

dev.off()


extract_test <- extract_test[-7]

nbS <- NaiveBayes(Sex ~ . , data = extract_test)
pdf("SampleGraph_testdata.pdf",width=7,height=5)
x=rnorm(100)
y=rnorm(100,5,1)
p<- plot(nbS)

dev.off()