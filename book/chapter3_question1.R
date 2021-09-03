library(tensorflow)
library(keras)

data(iris)
X = scale(iris[,1:4])
Y = iris[,5]
Y = keras::k_one_hot(as.integer(Y)-1L, 3)

model = keras_model_sequential()
model %>%
  layer_dense(units = 20L, activation = "relu", input_shape = list(4L)) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 3L, activation = "softmax")
# Softmax scales to (0, 1); 3 output nodes for 3 response classes/labels.
summary(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.5),
          metrics = c(metric_categorical_accuracy))

model_history =
  model %>%
  fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

#plot(model_history)

##########

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.005),
          metrics = c(metric_categorical_accuracy))

model_history2 =
  model %>%
  fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########  -> (very) low learning rate (may take (very) long (and may need very many epochs) and get stuck in local optima)

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.00001),
          metrics = c(metric_categorical_accuracy))

model_history3 =
  model %>%
  fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

keras::reset_states(model)

# Try ith higher epoch number
model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.00001),
          metrics = c(metric_categorical_accuracy))

model_history3 =
  model %>%
  fit(x = X, y = Y, epochs = 200L, batch_size = 20L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########  -> (Very) high learning rate (may skip optimum)

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 3),
          metrics = c(metric_categorical_accuracy))

model_history4 =
  model %>%
  fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########  -> Higher epoch number (possibly better fitting, maybe overfitting)

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 3),
          metrics = c(metric_categorical_accuracy))

model_history6 =
  model %>%
  fit(x = X, y = Y, epochs = 100L, batch_size = 20L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.5),
          metrics = c(metric_categorical_accuracy))

model_history7 =
  model %>%
  fit(x = X, y = Y, epochs = 100L, batch_size = 20L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.00001),
          metrics = c(metric_categorical_accuracy))

model_history8 =
  model %>%
  fit(x = X, y = Y, epochs = 100L, batch_size = 20L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########  -> Lower batch size

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 3),
          metrics = c(metric_categorical_accuracy))

model_history9 =
  model %>%
  fit(x = X, y = Y, epochs = 100L, batch_size = 5L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.5),
          metrics = c(metric_categorical_accuracy))

model_history10 =
  model %>%
  fit(x = X, y = Y, epochs = 100L, batch_size = 5L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.00001),
          metrics = c(metric_categorical_accuracy))

model_history11 =
  model %>%
  fit(x = X, y = Y, epochs = 100L, batch_size = 5L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########  -> Higher batch size (faster but less accurate)

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 3),
          metrics = c(metric_categorical_accuracy))

model_history12 =
  model %>%
  fit(x = X, y = Y, epochs = 100L, batch_size = 50L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.5),
          metrics = c(metric_categorical_accuracy))

model_history13 =
  model %>%
  fit(x = X, y = Y, epochs = 100L, batch_size = 50L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

##########

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.00001),
          metrics = c(metric_categorical_accuracy))

model_history14 =
  model %>%
  fit(x = X, y = Y, epochs = 100L, batch_size = 50L, shuffle = TRUE)

model %>%
  evaluate(X, Y)

####################
####################

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.05),
          metrics = c(metric_categorical_accuracy))

model_history15 =
  model %>%
  fit(x = X, y = Y, epochs = 150L, batch_size = 50L, shuffle = TRUE)

##########  -> shuffle = FALSE (some kind of overfitting)

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.05),
          metrics = c(metric_categorical_accuracy))

model_history16 =
  model %>%
  fit(x = X, y = Y, epochs = 150L, batch_size = 50L, shuffle = FALSE)

##########  -> shuffle = FALSE + lower batch size

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.05),
          metrics = c(metric_categorical_accuracy))

model_history17 =
  model %>%
  fit(x = X, y = Y, epochs = 150L, batch_size = 5L, shuffle = FALSE)

##########  -> shuffle = FALSE + higher batch size (many samples are taken at once, so no "hopping" any longer)

keras::reset_states(model)

model %>%
  compile(loss = loss_categorical_crossentropy, optimizer_adamax(lr = 0.05),
          metrics = c(metric_categorical_accuracy))

model_history18 =
  model %>%
  fit(x = X, y = Y, epochs = 150L, batch_size = 75L, shuffle = FALSE)


####################
####################

# Play around with the parameters on your own!

