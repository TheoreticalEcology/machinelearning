library(keras)
library(tensorflow)

load("Day2/Exoplanet.RData")

table(train$LABEL)

train = train[,-1]
test = test[,-1]

train$subset = "train"
test$subset = "test"
test$LABEL = NA

data = rbind(train, test)

data[,-c(1, ncol(data))] = scale(data[,-c(1, ncol(data))])

train = train[train$subset == "train", ]
test = test[train$subset == "test", ]

train = train[,-ncol(train)]

## Inner split:
indices = sample.int(nrow(train), 0.7*nrow(train))
sub_train = train[indices,]
sub_test = train[-indices,]
table(sub_train[,1])
table(sub_test[,1])

## Oversample
additional = sub_train[sub_train$LABEL == 1,]
sub_train = rbind(sub_train, addtional[sample.int(nrow(additional), 500L, replace = TRUE), ])
sub_train = sub_train[sample.int(nrow(sub_train), nrow(sub_train)), ]


library(keras)

## early stopping

# Deep Neural Networks and Regularization:
model = keras_model_sequential()
model %>% 
  layer_dense(units = 500L, activation = "relu", input_shape = ncol(train) -1L) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 500L, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 500L, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 500L, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1L, activation = "sigmoid")

summary(model)

model %>% 
  compile(loss = loss_binary_crossentropy, optimizer = optimizer_adamax(0.01))

early_stopping = callback_early_stopping(monitor = "val_loss", patience = 5L)

model_history = 
  model %>% 
    fit(x = as.matrix(sub_train[,-1]),
        y = as.matrix(sub_train[,1, drop = FALSE]), 
        class_weight = list("0" = 1, "1" = 1000),
        epochs = 10L, batch = 32L, shuffle = TRUE,
        validation_data = list(as.matrix(sub_test[,-1]), as.matrix(sub_test[,1,drop = FALSE])),
        callbacks = c(early_stopping))

plot(model_history)

pred = predict(model, as.matrix(sub_test[,-1]))
mean((ifelse(pred < 0.5, 0, 1) == sub_test[,1])[sub_test[,1] == 1])


# after 20 epochs the validation loss is increasing instead of decreasing!



## l1/l2
lambda = 0.001
model = keras_model_sequential()
model %>% 
  layer_dense(units = 100L, activation = "relu", input_shape = ncol(train) -2L, kernel_regularizer = regularizer_l1(lambda)) %>% 
  layer_dense(units = 100L, activation = "relu", kernel_regularizer = regularizer_l1(lambda)) %>% 
  layer_dense(units = 100L, activation = "relu", kernel_regularizer = regularizer_l1(lambda)) %>% 
  layer_dense(units = 2L, activation = "softmax", kernel_regularizer = regularizer_l1(lambda))

summary(model)

model %>% 
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adamax(0.01))


model_history_l1 = 
  model %>% 
  fit(x = train[,-c(1,2)], y = apply(train[,1:2],2L, as.integer), 
      epochs = 100L, 
      batch_size = 32L,
      verbose = 1L,
      validation_data = list(test[,-c(1,2)], apply(test[,1:2],2L, as.integer)))

plot(model_history_l1)


preds = predict(model, train[,-c(1,2)])


## How to set lambda?
## Dropout is a kind of model averaging
## For each sample we will put a binary mask of 0/1 on the nodes, setting randomly nodes with all its connection to 0.
## Thus, we train a indefinite set of sub networks



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
