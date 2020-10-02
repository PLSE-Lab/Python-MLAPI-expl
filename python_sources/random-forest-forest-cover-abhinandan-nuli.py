 #The environment is defined by the kaggle/rstats docker image

library(ggplot2) 
library(readr) 
library(readr)
library(randomForest)
library(reshape2)
library(pROC)
set.seed(123)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


myData=read.csv("../input/train.csv")
colnames(myData)



myData$Wilderness_Area1=as.factor(myData$Wilderness_Area1)
myData$Wilderness_Area2=as.factor(myData$Wilderness_Area2)
myData$Wilderness_Area3=as.factor(myData$Wilderness_Area3)
myData$Wilderness_Area4=as.factor(myData$Wilderness_Area4)
myData$Soil_Type1=as.factor(myData$Soil_Type1)
myData$Soil_Type2=as.factor(myData$Soil_Type2)
myData$Soil_Type3=as.factor(myData$Soil_Type3)
myData$Soil_Type4=as.factor(myData$Soil_Type4)
myData$Soil_Type5=as.factor(myData$Soil_Type5)
myData$Soil_Type6=as.factor(myData$Soil_Type6)
myData$Soil_Type7=as.factor(myData$Soil_Type7)
myData$Soil_Type8=as.factor(myData$Soil_Type8)
myData$Soil_Type9=as.factor(myData$Soil_Type9)
myData$Soil_Type10=as.factor(myData$Soil_Type10)
myData$Soil_Type11=as.factor(myData$Soil_Type11)
myData$Soil_Type12=as.factor(myData$Soil_Type12)
myData$Soil_Type13=as.factor(myData$Soil_Type13)
myData$Soil_Type14=as.factor(myData$Soil_Type14)
myData$Soil_Type15=as.factor(myData$Soil_Type15)
myData$Soil_Type16=as.factor(myData$Soil_Type16)
myData$Soil_Type17=as.factor(myData$Soil_Type17)
myData$Soil_Type18=as.factor(myData$Soil_Type18)
myData$Soil_Type19=as.factor(myData$Soil_Type19)
myData$Soil_Type20=as.factor(myData$Soil_Type20)
myData$Soil_Type21=as.factor(myData$Soil_Type21)
myData$Soil_Type22=as.factor(myData$Soil_Type22)
myData$Soil_Type23=as.factor(myData$Soil_Type23)
myData$Soil_Type24=as.factor(myData$Soil_Type24)
myData$Soil_Type25=as.factor(myData$Soil_Type25)
myData$Soil_Type26=as.factor(myData$Soil_Type26)
myData$Soil_Type27=as.factor(myData$Soil_Type27)
myData$Soil_Type28=as.factor(myData$Soil_Type28)
myData$Soil_Type29=as.factor(myData$Soil_Type29)
myData$Soil_Type30=as.factor(myData$Soil_Type30)
myData$Soil_Type31=as.factor(myData$Soil_Type31)
myData$Soil_Type32=as.factor(myData$Soil_Type32)
myData$Soil_Type33=as.factor(myData$Soil_Type33)
myData$Soil_Type34=as.factor(myData$Soil_Type34)
myData$Soil_Type35=as.factor(myData$Soil_Type35)
myData$Soil_Type36=as.factor(myData$Soil_Type36)
myData$Soil_Type37=as.factor(myData$Soil_Type37)
myData$Soil_Type38=as.factor(myData$Soil_Type38)
myData$Soil_Type39=as.factor(myData$Soil_Type39)
myData$Soil_Type40=as.factor(myData$Soil_Type40)
myData$Cover_Type=as.factor(myData$Cover_Type)
############################################
myData$Id=NULL
# remove zero variance
myData$Soil_Type7=NULL
myData$Soil_Type15=NULL
#############################################
#####################################
myData=myData[order(rnorm(nrow(myData))),]

a=0.8* nrow(myData)
r=nrow(myData)
#Split the data.....
myTrain=myData[1:a,]
myVal=myData[seq(a+1,r),]
a+1
length(seq(a+1,r))
#View(myTrain)
nrow(myVal)
nrow(myTrain)
############################
############################
str(myTrain$Cover_Type)
dim(myTrain)
head(myTrain,1)
summary(myTrain)
apply(myTrain, 2, function(x) length(unique(x)))
colnames(myTrain)
###################### Drawing Histogram ###################

d <- melt(myTrain[,c(1:10)])
ggplot(d,aes(x = value)) + 
  facet_wrap(~variable,scales = "free_x") + 
  geom_histogram()

#for (i in 11:54) myTrain[,i]=as.factor(myTrain[,i])

#myTrain[,11]=as.factor(myTrain[,11])

#myTrain$Wilderness_Area1=as.factor(myTrain$Wilderness_Area1)
#c=c(colnames(myTrain[11:54]))
#for (i in c) {
#  print(i)
#  myTrain[,i]=as.factor(myTrain[,i])
#}


#View(myTrain)
str(myTrain)
ncol(myTrain)-1
myRF=randomForest(Cover_Type~.,data=myTrain, type='classification', importance=TRUE, mtry=sqrt(ncol(myTrain)-1), ntree=20, sampsize= 100, nodesize= 25)
 
varImpPlot((myRF))

l=length(importance(myRF))
 

df=as.data.frame(importance(myRF))
#View(df)
df$var=rownames(df)
rownames(df)=NULL

opTrain=myTrain
remove=c(subset(df,df$IncNodePurity<= 0.00)$var)
remove=as.vector(remove)
for (i in remove){
  print(opTrain[,i])

opTrain[,i]=NULL
}

colnames(opTrain)
length(remove)
myRF2=randomForest(Cover_Type~.,data=opTrain, type='classification', mtry=sqrt(ncol(opTrain)-1), ntree=20, sampsize= 100, nodesize= 25)

varImpPlot(myRF2)
importance(myRF2)
myRF2

############ Prediction #########################


PredClass=predict(myRF, myVal, type="response", norm.votes = TRUE, proximity = FALSE)
length(PredClass)
t=table(Predictions=PredClass, Actual=myVal$Cover_Type)
dim(myVal)
dim(myTrain)
t
sum(diag(t))/sum(t)

PredProb=predict(myRF, myVal, type='prob')
auc=auc(myVal$Cover_Type, PredProb[,2])
auc
plot(roc(myVal$Cover_Type, PredProb[,2]))

############## FInd the best mtry #############

bestmtry=tuneRF(myTrain, myTrain$Cover_Type, ntreeTry = 100, stepFactor = 1, improve = 0.01
                , trace = T, plot = T, doBest=TRUE)
bestmtry


myRF3=randomForest(Cover_Type~.,data=myTrain, type='classification', importance=TRUE, mtry=15, ntree=100, sampsize= 100, nodesize= 25)
PredClass=predict(myRF3, myVal, type="response", norm.votes = TRUE, proximity = FALSE)
length(PredClass)
t=table(Predictions=PredClass, Actual=myVal$Cover_Type)
dim(myVal)
dim(myTrain)
t
sum(diag(t))/sum(t)

PredProb=predict(myRF3, myVal, type='prob')
auc=auc(myVal$Cover_Type, PredProb[,2])
auc
plot(roc(myVal$Cover_Type, PredProb[,2]))



# Any results you write to the current directory are saved as output.