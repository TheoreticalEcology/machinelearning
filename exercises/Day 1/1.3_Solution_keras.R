library(keras)
library(tensorflow)
tf$enable_eager_execution()
use_session_with_seed(42,disable_parallel_cpu = FALSE)

# Prepare data
X = scale(iris[,1:4])
Y = iris[,5]

# 1. Build model
model = keras_model_sequential()

model %>%
  layer_dense(units = 20L, activation = "relu", input_shape = list(4L)) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 3L, activation = "softmax") # softmax scales to 0 1 and overall to 0 - 1, 3 output nodes for 3 response classes/labels

summary(model)

# 2. Compile model - define loss and optimizer

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(0.001))


# 3. Fit model
Y = to_categorical(as.integer(Y)-1L, 3)
model_history =
  model %>%
    fit(x = X, y = apply(Y,2,as.integer), epochs = 30L, batch_size = 20L, shuffle = TRUE)

plot(model_history)

model %>%
  evaluate(X, Y)

predictions = predict(model, X) # probabilities for each class
preds = apply(predictions, 1, which.max)


oldpar = par()
par(mfrow = c(1,2))
plot(iris$Sepal.Length, iris$Petal.Length, col = iris$Species, main = "Observed")
plot(iris$Sepal.Length, iris$Petal.Length, col = preds, main = "Predicted")
par(oldpar)



##################

# Exercise 1 - Airquality with keras

# Prepare data
data = airquality
summary(data)
data = data[complete.cases(data$Ozone) & complete.cases(data$Solar.R),] # remove NAs
summary(data)


X = scale(data[,2:6])
Y = data[,1]

# 1. Build model
model = keras_model_sequential()

model %>%
  layer_dense(units = 20L, activation = "relu", input_shape = list(5L)) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 1L, activation = "linear") # one output dimension with a linear activation function

summary(model)

# 2. Compile model - define loss and optimizer

model %>%
  compile(loss = loss_mean_squared_error, optimizer_adamax(0.1))


# 3. Fit model
model_history =
  model %>%
  fit(x = X, y = matrix(Y, ncol = 1L), epochs = 30L, batch_size = 20L, shuffle = TRUE)

plot(model_history)

model %>%
  evaluate(X, Y)


# comparison against lm 
fit <- lm(Ozone ~ ., data = airquality)
mean(residuals(fit)^2)


predictions = predict(model, X)




plot(data$Ozone, predictions, main = "Predicted vs Observed")
abline(0, 1, col = "red")



# Bonus - TF Core

## functional keras API (see python TF help)
layers = tf$keras$layers
model = tf$keras$models$Sequential(
  c(
    layers$InputLayer(input_shape = list(5L)),
    layers$Dense(units = 20L, activation = tf$nn$relu),
    layers$Dense(units = 20L, activation = tf$nn$relu),
    layers$Dense(units = 20L, activation = tf$nn$relu),
    layers$Dense(units = 1L, activation = NULL)
  )
)

epochs = 200L
optimizer = tf$keras$optimizers$Adamax(0.01)

# Stochastic gradient optimization is more efficient
get_batch = function(batch_size = 32L){
  indices = sample.int(nrow(X), size = batch_size)
  return(list(bX = X[indices,], bY = Y[indices]))
}

steps = floor(nrow(X)/32) * epochs # we need nrow(X)/32 steps for each epoch

for(i in 1:steps){
  batch = get_batch()
  bX = tf$constant(batch$bX)
  bY = tf$constant(matrix(batch$bY, ncol = 1L))

  # Automatic diff
  with(tf$GradientTape() %as% tape, {
    pred = model(bX) # we record the operation for our model weights
    loss = tf$reduce_mean(tf$keras$losses$mean_squared_error(bY, pred))
  })

  gradients = tape$gradient(loss, model$weights) # calculate the gradients for the loss at our model$weights / backpropagation

  optimizer$apply_gradients(purrr::transpose(list(gradients, model$weights))) # update our model weights with the above specified learning rate

  if(i %% floor(nrow(X)/32)*20 == 0) cat("Loss: ", loss$numpy(), "\n") # print loss every 20 epochs
}





# Solution for iris:
# functional keras API (see python TF help)
X = scale(iris[,1:4])
Y = iris[,5]
Y = keras::to_categorical(as.integer(Y)-1L, 3)


layers = tf$keras$layers
model = tf$keras$models$Sequential(
  c(
    layers$InputLayer(input_shape = list(4L)),
    layers$Dense(units = 20L, activation = tf$nn$relu),
    layers$Dense(units = 20L, activation = tf$nn$relu),
    layers$Dense(units = 20L, activation = tf$nn$relu),
    layers$Dense(units = 3L, activation = tf$nn$softmax)
  )
)

epochs = 200L
optimizer = tf$keras$optimizers$Adamax(0.01)

# Stochastic gradient optimization is more efficient
get_batch = function(batch_size = 32L){
  indices = sample.int(nrow(X), size = batch_size)
  return(list(bX = X[indices,], bY = Y[indices,]))
}

steps = floor(nrow(X)/32) * epochs # we need nrow(X)/32 steps for each epoch

for(i in 1:steps){
  batch = get_batch()
  bX = tf$constant(batch$bX)
  bY = tf$constant(batch$bY)

  # Automatic diff
  with(tf$GradientTape() %as% tape, {
    pred = model(bX) # we record the operation for our model weights
    loss = tf$reduce_mean(tf$keras$losses$categorical_crossentropy(bY, pred))
  })

  gradients = tape$gradient(loss, model$weights) # calculate the gradients for the loss at our model$weights / backpropagation

  optimizer$apply_gradients(purrr::transpose(list(gradients, model$weights))) # update our model weights with the above specified learning rate

  if(i %% floor(nrow(X)/32)*20 == 0) cat("Loss: ", loss$numpy(), "\n") # print loss every 20 epochs
}


