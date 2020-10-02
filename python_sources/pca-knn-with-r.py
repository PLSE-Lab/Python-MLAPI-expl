library(readr)
library(class)

train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")

cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

cat("Start PCA\n")
train.columns.var <- apply(train[,-1], 2, var)
train.zeroVarRemoved <- train[,c(F, train.columns.var!=0)]
pca.result <- prcomp(train.zeroVarRemoved, scale=T)
train.pca <- pca.result$x
test.pca <- predict(pca.result, test)

cat("Start KNN\n")
set.seed(0)
numTrain <- 30000
rows <- sample(1:nrow(train), numTrain)
train.col.used <- 1:43
prediction <- knn(train.pca[rows,train.col.used], test.pca[,train.col.used], train[rows,1], k=3)
prediction.table <- data.frame(ImageId=1:nrow(test), Label=prediction)
write_csv(prediction.table, "pca_knn.csv")

