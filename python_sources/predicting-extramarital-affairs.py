#!/usr/bin/env python
# coding: utf-8

# In[ ]:


library(Ecdat)
library(randomForest)
library(dplyr)
library(reldist)
set.seed(734209672)

# sex,age,ym,child,religious,education,occupation,rate,nbaffairs
# male,37,10,no,3,18,7,4,0
# female,27,4,no,4,14,6,4,0
# female,32,15,yes,1,12,1,4,0
# male,57,15,yes,5,18,6,5,0
# male,22,0.75,no,2,17,6,3,0

# Split dataset
X <- Fair[sample(nrow(Fair)),]
train <- X[1:400,]
test <- X[401:601,]

# Random forest
rf <- randomForest(nbaffairs ~ ., data = train, ntree = 200, importance = T)
results <- data.frame(actual = test$nbaffairs, expected = predict(rf, test))

# Variable importance plot
varImpPlot(rf)

# Gini
SumModelGini <- function(solution, submission) {
  df = data.frame(solution = solution, submission = submission)
  df <- df[order(df$submission, decreasing = TRUE),]
  df$random = (1:nrow(df))/nrow(df)
  totalPos <- sum(df$solution)
  df$cumPosFound <- cumsum(df$solution) 
  df$Lorentz <- df$cumPosFound / totalPos
  df$Gini <- df$Lorentz - df$random
  return(sum(df$Gini))
}
NormalizedGini <- function(solution, submission) {
  SumModelGini(solution, submission) / SumModelGini(solution, solution)
}

print(NormalizedGini(results$actual, results$expected))

