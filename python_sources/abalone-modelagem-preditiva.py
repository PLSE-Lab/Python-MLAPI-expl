## Importar as bibliotecas necessárias
# Vide: https://cran.r-project.org/web/packages/caret/
library(caret)

## Carregar os dados de treino
train_data <- read.csv(file = "../input/abalone-train.csv", header = TRUE, sep = ",", row.names = "id")
head(train_data, 5)
str(train_data)

## Encontrar e remover possíveis outliers
# outliers = train_data[
#   (train_data$height < 0.01 | train_data$height > 0.3) |
#     (train_data$viscera_weight < 0.0001 | train_data$viscera_weight > 0.6),]
# outliers
train_data <- train_data[! ((train_data$height < 0.01 | train_data$height > 0.3) |
    (train_data$viscera_weight < 0.0001 | train_data$viscera_weight > 0.6)),]
str(train_data)

### Definir dados de entrada
X = train_data[,1:8]
y = train_data$rings
str(X)
head(X)
head(y)

## Definir algoritmos a serem usados
set.seed(42)
methods = c("rpart", "rf", "gbm", "treebag", "glm", "lm")
models = list()

## Treinar modelos com diferentes algoritmos
for (method in methods) {
#method="lm"
#{
  print(paste("Treinando o algoritmo", toupper(method), "..."))
  
  # K-Fold cross-validation
  train_control <- trainControl(method="cv", number=10)
  
  # treinar o modelo preditivo
  model <- train(x = X,
                 y = y,
                 method = method,
                 trControl = train_control,
                 metric = "RMSE")
  print(model)
  models[[method]] <- model
}

## Exibir os resultados de cada algoritmo
# Vide: https://machinelearningmastery.com/machine-learning-evaluation-metrics-in-r/
results <- resamples(models)
summary(results)
dotplot(results)

## Carregar os dados de avaliação (teste)
test_data <- read.csv(file = "../input/abalone-test.csv", header = TRUE, sep = ",", row.names = "id")
str(test_data)
head(test_data, 5)

## Gerar arquivos de envio para todos os algoritmos selecionados
for (method in names(models)) {
  model <- models[method]
  print(paste("Gerando arquivo para o algoritmo", toupper(method), "..."))
  
  # Prever os resultados usando o modelo já treinado
  y_pred <- predict(object = model, newdata = test_data)
  names(y_pred) <- c("rings")
  
  # Preparar o arquivo de envio
  submission <- data.frame("id" = row.names(test_data), y_pred)
  write.csv(submission,
            file = paste("submissions/abalone-submission-r-", method, ".csv", sep = ""), 
            quote = FALSE,
            row.names = FALSE)
}