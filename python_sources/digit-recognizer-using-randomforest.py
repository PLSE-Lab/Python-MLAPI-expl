# You can write R code here and then click "Run" to run it on our platform

library(readr)

# The competition datafiles are in the directory ../input
# Read competition data files:
train <- read_csv("../input/train.csv")
test <- read_csv("../input/test.csv")

# Write to the log:
cat(sprintf("Training set has %d rows and %d columns\n", nrow(train), ncol(train)))
cat(sprintf("Test set has %d rows and %d columns\n", nrow(test), ncol(test)))

# Generate output files with write_csv(), plot() or ggplot()
# Any files you write to the current directory get shown as outputs
library(caTools)
train$label = as.factor(train$label)
split = sample.split(train$label,SplitRatio = 0.7)
tr = subset(train,split == T)
te = subset(train,split == F)
library(h2o)
localh20 = h2o.init(max_mem_size = '6g',nthreads = -1)
train_h20 = as.h2o(train)
test_h20 = as.h2o(test)
tr_h2o = as.h2o(tr)
te_h2o = as.h2o(te)
s <- proc.time()
RF = h2o.randomForest(x = 2:785, y = 1,training_frame = tr_h2o)
s - proc.time()

h2o.performance(RF)
RF.predict = h2o.predict(RF,newdata = te_h2o)
outcomes = as.vector(as.factor(as.numeric(RF.predict$predict)))
length(outcomes)
table(te$label,outcomes)

RF.final = h2o.randomForest(x = 2:785, y = 1,training_frame = train_h20)
RF.final.predict = h2o.predict(RF.final,newdata = test_h20)
out = as.vector(as.numeric(RF.final.predict$predict))
id = seq(1:28000)
submiss = data.frame(id,out,row.names = NULL)
head(submiss)
colnames(submiss) = c("ImageId","Label")
write.csv(submiss,"RFsub.csv",row.names = F)
