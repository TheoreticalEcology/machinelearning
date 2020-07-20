# Introduction to keras
## **** Objective of this lesson: familiarize yourself with keras ****

## Background:
## keras is a higher level API within TF and developed to build easily neural networks
## keras can be found within TF: tf$keras...
## however, the rstudio team built a pkg on top of tf$keras:
library(keras)
library(tensorflow)
data(iris)
head(iris)

## Johannes: Sag erstmal welche Daten sie skalieren sollen usw. 
## steht nirgendwo davor und ich dachte mir gerade, was will er jetzt 
## und sag noch was die response sein soll 

## Prepare data
## Scale data: centering+standardization, see ?scale
X = scale(iris[,1:4])
Y = iris[,5]

## keras/tf cannot handle factors and we have to create contrasts (one-hot encoding):
## Anmerkung JOhannes: vielleicht das -1L erklären.. im endeffekt ist doch die 
## Logik, dass arrays start at zero in python.. glaub das wissen fast keine Leute 
## im Kurs (vielleicht auch sagen woher das kommt (pointet auf den Abstand zwischen dem ersten 
## Element im Ram und dem aktuellen), aber macht bei python eigentlichen keinen Sinn, keine Ahnung )

Y = to_categorical(as.integer(Y)-1L, 3)
Y # 3 colums, one for each level in the response

## 1. Build model 
## Johannes: welches Modell sollen wir builden ?
## Allgemein: kennen sie den %>% operator? 
## Ich glaube es wäre einfacher direkte anweisungen zu geben ala: 
## 1. Initiliaze a sequential model in keras
## 2. add layers with the properties that ... 
## 3. compile the model with a cross entropy lossfunction 
## 4. fit the model to the old data 

model = keras_model_sequential()

## Add hidden layers (we will learn more about DNNs during the next days)
model %>%
  layer_dense(units = 20L, activation = "relu", input_shape = list(4L)) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 3L, activation = "softmax") # softmax scales to 0 1 and overall to 0 - 1, 
## 3 output nodes for 3 response classes/labels

summary(model)

## 2. Compile model - define loss and optimizer

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(0.001))


## 3. Fit model
model_history =
  model %>%
    fit(x = X, y = apply(Y,2,as.integer), epochs = 30L, batch_size = 20L, shuffle = TRUE)

plot(model_history)

model %>%
  evaluate(X, Y)

predictions = predict(model, X) # probabilities for each class
predictions # quasi-probabilities for each species
preds = apply(predictions, 1, which.max) # in each row, extract the species with the highest probability

## Accuracy:
mean(preds == as.integer(iris$Species))

oldpar = par()
par(mfrow = c(1,2))
plot(iris$Sepal.Length, iris$Petal.Length, col = iris$Species, main = "Observed")
plot(iris$Sepal.Length, iris$Petal.Length, col = preds, main = "Predicted") # Can you see the errors?
par(oldpar)






# Exercise - Airquality with keras
## Prepare data
data = airquality

### have a look at the summary
summary(data)

### remove NAs! Keras cannot handle NAs! If you do not know how to remove NAs, use google 
### (e.g. with the query: remove-rows-with-all-or-some-nas-missing-values-in-data-frame)
data = data[complete.cases(data),] # remove NAs
summary(data)

## Johannes: hier sagen was sie überhaupt fitten sollen: also Ozone
### scale data and create X and Y
X = scale(data[,2:6])
Y = data[,1]

### 1. Build model
### Warning: we have now a regression task, use the following layer as last layer in your model:
### layer_dense(units = 1L, activation = "linear") 
model = keras_model_sequential()

model %>%
  layer_dense(units = 20L, activation = "relu", input_shape = list(5L)) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 1L, activation = "linear") # one output dimension with a linear activation function

summary(model)

### 2. Compile model - define loss and optimizer
### For regression, we have also to change the loss function, use loss_mean_squared_error (not as string, i.e. in "loss_mean...")

model %>%
  compile(loss = loss_mean_squared_error, optimizer_adamax(0.1))


### 3. Fit model
### Tip: only matrices are accepted for x and y by keras. R often drops an one column matrix into a vector (change it back to a matrix!)
model_history =
  model %>%
  fit(x = X, y = matrix(Y, ncol = 1L), epochs = 30L, batch_size = 20L, shuffle = TRUE)

plot(model_history)

model %>%
  evaluate(X, Y)

### root mean squared error:
### create predictions with predict(model, X)
### calculate RMSE (use google if you do not know the equation (however, name == equation))
pred_keras = predict(model, X)
rmse_keras = mean(sqrt((Y - pred_keras)^2))

## comparison against lm
### fit a lm(...)  
fit = lm(Ozone ~ ., data = data)
### create predictions with predict(model_lm, data)
pred_lm = predict(fit, data)
rmse_lm = mean(sqrt((Y - pred_lm)^2))

### compare the RMSEs, which model is better?


### plot predicitions against true values:
plot(data$Ozone, pred_keras, main = "Predicted vs Observed", col = "red")
points(data$Ozone, pred_lm, col = "blue")


## Bonus - TF Core
### functional keras API (see python TF help)
### airquality example, go through the code line by line and try to understand it
layers = tf$keras$layers
model = tf$keras$models$Sequential(
  c(
    layers$InputLayer(input_shape = list(5L)),
    layers$Dense(units = 20L, activation = tf$nn$relu),
    layers$Dense(units = 20L, activation = tf$nn$relu),
    layers$Dense(units = 20L, activation = tf$nn$relu),
    layers$Dense(units = 1L, activation = NULL) # no activation == "linear" 
  )
)

epochs = 200L
optimizer = tf$keras$optimizers$Adamax(0.01)

### Stochastic gradient optimization is more efficient
### in each optimization step, we use a random subset of the data
get_batch = function(batch_size = 32L){
  indices = sample.int(nrow(X), size = batch_size)
  return(list(bX = X[indices,], bY = Y[indices]))
}
get_batch()

steps = floor(nrow(X)/32) * epochs # we need nrow(X)/32 steps for each epoch


for(i in 1:steps){
  # get data
  batch = get_batch()
  
  # transform it into tensors
  bX = tf$constant(batch$bX)
  bY = tf$constant(matrix(batch$bY, ncol = 1L))

  # Automatic diff:
  # record computations with respect to our model variables
  with(tf$GradientTape() %as% tape, {
    pred = model(bX) # we record the operation for our model weights
    loss = tf$reduce_mean(tf$keras$losses$mean_squared_error(bY, pred))
  })

  # calculate the gradients for our model$weights at the loss / backpropagation
  gradients = tape$gradient(loss, model$weights) 
  
  # update our model weights with the above specified learning rate
  optimizer$apply_gradients(purrr::transpose(list(gradients, model$weights))) 
  
  if(i %% floor(nrow(X)/32)*20 == 0) cat("Loss: ", loss$numpy(), "\n") # print loss every 20 epochs
}


## Bonus exercise - change the code from above for the iris dataset
### Tip: in tf$keras$losses$... you can find various loss functions 
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


