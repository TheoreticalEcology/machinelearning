library(keras)
library(tensorflow)
tf$enable_eager_execution()


# simple optimization f(x) = x^2

func = function(x) return(x^2)
opt = optim(1.0, func)
opt$par


# linear regression f(x) = x * w, w = parameter/coefficients
data = airquality[complete.cases(airquality$Ozone) & complete.cases(airquality$Solar.R),]
X = scale(data[,-1])
Y = data$Ozone

linear_regression = function(w) {
  pred = w[1]*X[,1] + # Solar.R
         w[2]*X[,2] + # Wind
         w[3]*X[,3] + # Temp
         w[4]*X[,4] + # Month
         w[5]*X[,5] +
         w[6]         # or X %*% w[1:5] + w[6]
  
  # loss  = MSE, we want to find the optimal weights to minimize the sum of squared residuals
  loss = mean((pred - Y)^2)
  return(loss)
}

linear_regression(runif(6))

# bruteforce:
random_search = matrix(runif(6*5000,-10,10), 5000, 6)
losses = apply(random_search, 1, linear_regression)
plot(losses, type = "l")
random_search[which.min(losses),]

opt = optim(runif(6, -1, 1), linear_regression)
opt$par

coef(lm(Y~X))

# Exercise 1 
# try the different optimizer in optim and compare the results with coef(lm(Y~X))





# Bonus TF-Core
dtype = "float64"
W = tf$Variable(tf$constant(runif(5,-1,1), dtype, shape = list(5L, 1L)))
B = tf$Variable(tf$constant(runif(1, -1, 1), dtype))
optimizer = tf$contrib$optimizer_v2$AdamOptimizer(0.01)

epochs = 200L

get_batch = function(batch_size = 32L){
  indices = sample.int(nrow(X), size = batch_size)
  return(list(bX = X[indices,], bY = Y[indices]))
}

steps = floor(nrow(X)/32) * epochs

for(i in 1:steps){
  
  batch = get_batch()
  bX = tf$constant(batch$bX, dtype)
  bY = tf$constant(matrix(batch$bY, ncol = 1L), dtype)
  
  with(tf$GradientTape() %as% tape, {
    pred = tf$matmul(tf$constant(bX, dtype), W) + B
    loss = tf$reduce_mean(tf$square(pred - tf$constant(bY, dtype)))
  })
  
  gradients = tape$gradient(loss, c(W, B))
  
  optimizer$apply_gradients(purrr::transpose(list(gradients, c(W, B))))
  
  if(i %% floor(nrow(X)/32)*20 == 0) cat("Loss: ", loss$numpy(), "\n")  
}







# NN have complex loss surfaces.
# Too high learning rates and optima might be jumped over... 
# Too low learning rates and we might be land in local optima or it might take too long..


X = scale(iris[,1:4])
Y = iris[,5]
Y = keras::k_one_hot(as.integer(Y)-1L, 3)

model = keras_model_sequential()

model %>%
  layer_dense(units = 20L, activation = "relu", input_shape = list(4L)) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 3L, activation = "softmax") # softmax scales to 0 1 and overall to 0 - 1, 3 output nodes for 3 response classes/labels

summary(model)


model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.5), metrics = c(metric_categorical_accuracy))

model_history =
  model %>%
    fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

model %>%
  evaluate(X, Y)
plot(model_history)


keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.00005), metrics = c(metric_categorical_accuracy))
model_history2 =
  model %>%
   fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)
plot(model_history2)
model %>%
  evaluate(X, Y)


# Exercise 2 
# play around with the learning_rate (we want to increase the categorical_accuracy)





