library(keras)
library(tensorflow)
library(tidyverse)
library(missRanger)
library(caret)
library(Metrics)

load("datasets.RData")
train = nasa$train
test = nasa$test
table(train$Hazardous)


train$subset = "train"
test$subset = "test"
test$Hazardous = NA

data = rbind(train, test)

## Explore and clean data
str(data)
summary(data)
# Remove Equinox
data = data %>% select(-Equinox, -Orbiting.Body, -Orbit.Determination.Date, -Close.Approach.Date)


# Impute missing values using a randomForest
data_impute = data %>% select(-Hazardous)
imputed = missRanger::missRanger(data_impute, maxiter = 1L, num.trees = 20L)

# scale data
data = cbind( data %>% select(Hazardous), scale(imputed %>% select(-subset)), data.frame(subset = data$subset))
summary(data)


## Outer split
train = data[data$subset == "train", ]
test = data[data$subset == "test", ]

train = train %>% select(-subset)
test = test %>% select(-subset)


## 10-Fold CV:
cv_indices = caret::createFolds(train$Hazardous, k = 10)
result = matrix(NA, 10L, 2L)
colnames(result) = c("train_auc", "test_auc")

for(i in 1:10) {
  indices = cv_indices[[i]]
  sub_train = train[-indices,]
  sub_test = train[indices,]

  
  # Deep Neural Networks and Regularization:
  model = keras_model_sequential()
  model %>% 
    layer_dense(units = 100L, activation = "relu", input_shape = ncol(sub_train) -1L) %>% 
    #layer_dropout(rate = 0.3) %>% 
    layer_dense(units = 100L, activation = "relu") %>% 
    #layer_dropout(rate = 0.3) %>% 
    layer_dense(units = 1L, activation = "sigmoid")
  
  #early_stopping = callback_early_stopping(monitor = "val_loss", patience = 5L)
  
  summary(model)
  
  model %>% 
    compile(loss = loss_binary_crossentropy, optimizer = optimizer_adamax(0.01))
  
  model_history = 
    model %>% 
      fit(x = as.matrix(sub_train %>% select(-Hazardous)),
          y = as.matrix(sub_train%>% select(Hazardous)), 
          #callbacks=c(early_stopping),
          #validation_split = 0.2,
          epochs = 30L, batch = 32L, shuffle = TRUE)
  
  plot(model_history)
  pred_train = predict(model, as.matrix(sub_train %>% select(-Hazardous)))
  pred_test = predict(model, as.matrix(sub_test %>% select(-Hazardous)))
  result[i, 1] = Metrics::auc(sub_train$Hazardous, pred_train)
  result[i, 2] = Metrics::auc(sub_test$Hazardous, pred_test)
}
print(result)
## The model setup seems to be fine, train and predict for outer validation split:
model = keras_model_sequential()
model %>% 
  layer_dense(units = 50L, activation = "relu", input_shape = ncol(sub_train) -1L) %>% 
  #layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 50L, activation = "relu") %>% 
  #layer_dropout(rate = 0.3) %>% 
  layer_dense(units = 1L, activation = "sigmoid")

#early_stopping = callback_early_stopping(monitor = "val_loss", patience = 5L)

summary(model)

model %>% 
  compile(loss = loss_binary_crossentropy, optimizer = optimizer_adamax(0.01))

model_history = 
  model %>% 
  fit(x = as.matrix(train %>% select(-Hazardous)),
      y = as.matrix(train%>% select(Hazardous)), 
      #callbacks=c(early_stopping),
      #validation_split = 0.2,
      epochs = 30L, batch = 32L, shuffle = TRUE)

preds = predict(model, as.matrix(test[,-1]))
write.csv(data.frame(y=ifelse(preds < 0.5, 0, 1)), file = "max_1.csv")





## Dropout, the paper suggest 0.5 dropout rate 
model = keras_model_sequential()
model %>% 
  layer_dense(units = 100L, activation = "relu", input_shape = ncol(train) -2L) %>% 
  layer_dropout(0.5) %>% 
  layer_dense(units = 100L, activation = "relu") %>% 
  layer_dropout(0.5) %>% 
  layer_dense(units = 100L, activation = "relu") %>%
  layer_dropout(0.5) %>% 
  layer_dense(units = 2L, activation = "softmax")

summary(model)

model %>% 
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adamax(0.01))

# One big advantage is, that we do not need to control the epochs

model_history_d = 
  model %>% 
  fit(x = train[,-c(1,2)], y = apply(train[,1:2],2L, as.integer), 
      epochs = 100L, 
      batch_size = 32L,
      verbose = 1L,
      validation_data = list(test[,-c(1,2)], apply(test[,1:2],2L, as.integer)))

plot(model_history_d)



## Early stopping
### Why not stop training when val loss starts to increase? 
### On Monday, the validation loss started to increase after 20 epochs, why not stop there automatically?

model = keras_model_sequential()
model %>% 
  layer_dense(units = 100L, activation = "relu", input_shape = ncol(train) -2L) %>% 
  layer_dense(units = 100L, activation = "relu") %>% 
  layer_dense(units = 100L, activation = "relu") %>% 
  layer_dense(units = 2L, activation = "softmax")

summary(model)

early = callback_early_stopping(patience = 10L)

model %>% 
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adamax(0.01))


model_history_early = 
  model %>% 
  fit(x = train[,-c(1,2)], y = apply(train[,1:2],2L, as.integer), 
      epochs = 100L, 
      batch_size = 32L,
      verbose = 1L,
      validation_data = list(test[,-c(1,2)], apply(test[,1:2],2L, as.integer)),
      callbacks = c(early))

plot(model_history_early)
# after 20 epochs the validation loss is increasing instead of decreasing!



## Exercise - 1
## - train a DNN with a lot of hidden units on your projects
## - can you improve your previous results?
