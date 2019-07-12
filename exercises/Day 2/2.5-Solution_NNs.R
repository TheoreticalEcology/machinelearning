library(keras)
library(tensorflow)
tf$enable_eager_execution()

# Classification

data = iris
data[,1:4] = apply(data[,1:4],2, scale)

indices = sample.int(nrow(data), 0.7*nrow(data))
train = data[indices,]
test = data[-indices,]

model = keras_model_sequential()

model %>% 
  layer_dense(units = 20L, 
              activation = tf$nn$relu, 
              use_bias = TRUE, 
              input_shape = c(4L)) %>%
  layer_dense(units = 3L, activation = tf$nn$softmax)
# The soft transform logits into probabilities and the sum to 1
# y_i = e^y_i / sum e^y_j

summary(model)

model %>% 
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adamax(0.001))

model %>% 
  fit(x = as.matrix(train[,1:4]), y = k_one_hot(as.integer(train$Species) -1L, 3L)$numpy(), epochs = 50L, schuffle = TRUE)

pred = predict(model, as.matrix(train[,1:4]))  
# probabilities for classes sum up to 1
apply(pred, 1, sum)


## Exercise - 1
## - change activation function to sigmoid
## - check predictions

model = keras_model_sequential()

model %>% 
  layer_dense(units = 20L, 
              activation = tf$nn$relu, 
              use_bias = TRUE, 
              input_shape = c(4L)) %>%
  layer_dense(units = 3L, activation = tf$nn$sigmoid)
# The soft transform logits into probabilities and the sum to 1
# y_i = e^y_i / sum e^y_j

summary(model)

model %>% 
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adamax(0.001))

model %>% 
  fit(x = as.matrix(train[,1:4]), y = k_one_hot(as.integer(train$Species) -1L, 3L)$numpy(), epochs = 50L, schuffle = TRUE)

pred = predict(model, as.matrix(train[,1:4]))  
# probabilities for classes sum up to 1
apply(pred, 1, sum)


## Exercise - 2
## - split airquality into train and test
## - build model. Which activation function in the last layer?
## - add l1/l2 loss 

data = airquality[complete.cases(airquality$Ozone) & complete.cases(airquality$Solar.R),]
data[,2:6] = apply(data[,2:6],2, scale)

indices = sample.int(nrow(data), 0.7*nrow(data))
train = data[indices,]
test = data[-indices,]

model = keras_model_sequential()

model %>% 
  layer_dense(units = 20L, 
              activation = tf$nn$relu, 
              use_bias = TRUE, 
              input_shape = c(5L),
              kernel_regularizer = regularizer_l1(10.0)) %>%
  layer_dense(units = 20L, 
              activation = tf$nn$relu,
              kernel_regularizer = regularizer_l1(10.0)) %>% 
  layer_dense(units = 1L, activation = "linear")

model %>% 
  compile(loss = loss_mean_squared_error, optimizer = optimizer_adamax(0.1))

model %>% 
  fit(x = as.matrix(train[,2:6]), y = matrix(train[,1], ncol = 1L), epochs = 100L, schuffle = TRUE)

pred = predict(model, as.matrix(test[,2:6]))
plot(test$Temp, test$Ozone)
lines(test$Temp[order(test$Temp)], pred[order(test$Temp)], col = "red")
