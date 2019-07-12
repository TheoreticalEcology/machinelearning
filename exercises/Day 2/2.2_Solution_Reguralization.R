library(keras)
library(tensorflow)
tf$enable_eager_execution()


data = airquality[complete.cases(airquality),]
X = scale(data[,-1])
Y = data$Ozone


# l1/l2 on linear model
model = keras_model_sequential()

## unconstrained
model %>%
  layer_dense(units = 1L, activation = "linear", input_shape = list(dim(X)[2]))
summary(model)

model %>%
  compile(loss = loss_mean_squared_error, optimizer_adamax(lr = 0.5), metrics = c(metric_mean_squared_error))

model_history =
  model %>%
  fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

unconstrained = model$get_weights()
summary(lm(Y~X))
coef(lm(Y~X))




## l1
model = keras_model_sequential()

model %>%
  layer_dense(units = 1L, activation = "linear", input_shape = list(dim(X)[2]), kernel_regularizer = regularizer_l1(10), bias_regularizer = regularizer_l1(10))
summary(model)

model %>%
  compile(loss = loss_mean_squared_error, optimizer_adamax(lr = 0.5), metrics = c(metric_mean_squared_error))

model_history =
  model %>%
  fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

l1 = model$get_weights()
summary(lm(Y~X))
coef(lm(Y~X))

cbind(unlist(l1), unlist(unconstrained))


# Exercise 1 
# - put l2 on weights
# - put l1 and l2 on weights

## l2
model = keras_model_sequential()

model %>%
  layer_dense(units = 1L, activation = "linear", input_shape = list(dim(X)[2]), kernel_regularizer = regularizer_l2(10), bias_regularizer = regularizer_l2(10))
summary(model)

model %>%
  compile(loss = loss_mean_squared_error, optimizer_adamax(lr = 0.5), metrics = c(metric_mean_squared_error))

model_history =
  model %>%
  fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

l2 = model$get_weights()
summary(lm(Y~X))
coef(lm(Y~X))

cbind(unlist(l2), unlist(l1), unlist(unconstrained))


## l1+l2
model = keras_model_sequential()

model %>%
  layer_dense(units = 1L, activation = "linear", input_shape = list(dim(X)[2]), kernel_regularizer = regularizer_l1_l2(10, 10), bias_regularizer = regularizer_l1_l2(10, 10))
summary(model)

model %>%
  compile(loss = loss_mean_squared_error, optimizer_adamax(lr = 0.5), metrics = c(metric_mean_squared_error))

model_history =
  model %>%
  fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

l1_l2 = model$get_weights()
summary(lm(Y~X))
coef(lm(Y~X))

cbind(unlist(l1_l2), unlist(l2), unlist(l1), unlist(unconstrained))


# Bonus
# - try to understand the following tensorflow core code
# - implement l1 and l2 in tensorflow core
# - implement elastic net in tensorflow core (lambda * ( (1-alpha)/2*l2 + alpha*l1 ))
data = airquality[complete.cases(airquality$Ozone) & complete.cases(airquality$Solar.R),]
X = scale(data[,-1])
Y = data$Ozone


W = tf$Variable(
  tf$constant(runif(ncol(X), -1, 1), shape = list(ncol(X), 1L), "float64")
)
B = tf$Variable(tf$constant(runif(1,-1,1), "float64"))

epochs = 200L
optimizer = tf$keras$optimizers$Adamax(1)

get_batch = function(batch_size = 32L){
  indices = sample.int(nrow(X), size = batch_size)
  return(list(bX = tf$constant(X[indices,], "float64"), bY = tf$constant(Y[indices], "float64", list(batch_size, 1L))))
}

steps = floor(nrow(X)/32) * epochs

zero = tf$constant(0.0, "float64")
one = tf$constant(1.0, "float64")
two = tf$constant(2.0, "float64")

l1_tf = function(W, B, lambda = tf$constant(1.0, "float64")) lambda * tf$reduce_mean(tf$abs(W) + tf$abs(B))
l2_tf = function(W, B, lambda = tf$constant(1.0, "float64")) lambda * tf$reduce_mean(tf$square(W) + tf$square(B))
elastic_tf = function(W, B, alpha = tf$constant(0.5, "float64"),lambda = tf$constant(1.0, "float64")) {
  lambda * ((one - alpha)/two * l2_tf(W, B, one) + alpha* l1_tf(W, B, one))
}

for(i in 1:steps){
  batch = get_batch()
  bX = batch$bX
  bY = batch$bY
  
  with(tf$GradientTape() %as% tape,{
    pred = tf$matmul(bX, W) + B
    loss = tf$reduce_mean(tf$keras$losses$mean_squared_error(bY, pred)) + elastic_tf(W, B)
  })
  
  gradients = tape$gradient(loss, c(W, B))
  
  optimizer$apply_gradients(purrr::transpose(list(gradients, c(W, B))))
  
  if(i %% floor(nrow(X)/32)*20 == 0) cat("Loss: ", loss$numpy(), "\n")
  
}

