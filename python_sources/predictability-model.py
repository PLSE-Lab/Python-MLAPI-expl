Mushrooms Predictability Model
```{r,eval=TRUE,echo=TRUE,warning=FALSE, message=FALSE}

library("ggplot2")
library("class")
library("useful")
library("randomForest")

Mushrooms.database <- read.csv("../input/mushrooms.csv", header = T)
set.seed(1000)
# Spliting up the data into training (70%) and testing (30%)
data <- sample(2, nrow(Mushrooms.database), replace = T, prob = c(0.7, 0.3))
traindata <- Mushrooms.database[data == 1,]
testdata <- Mushrooms.database[data == 2,]
# Tree functions
variablesTree <- class ~ cap.shape + cap.surface + cap.color + bruises + odor + gill.attachment + gill.spacing + gill.size + gill.color + stalk.shape + stalk.root + stalk.surface.above.ring + stalk.surface.below.ring + stalk.color.above.ring + stalk.color.below.ring + veil.type + veil.color + ring.number + ring.type + spore.print.color + population + habitat
# Applying the algarithm
treeRF <- randomForest(variablesTree, data = traindata, ntree=100, proximity = T)
# Importance of each variable
dataimp <- varImpPlot(treeRF, main = "Importance of each variable")
# Evolution of the error according to the number of trees
plot(treeRF, main ="Evolution of the error")
# Capacity for prediction of the model
testPredRF <- predict(treeRF, newdata = testdata)
prediction_table <- table(testPredRF, testdata$class)
print(prediction_table)
# Model Reliability (Result = 100%)
predictability <- sum(testPredRF == testdata$class)/ length(testdata$class)*100
print(predictability)
```