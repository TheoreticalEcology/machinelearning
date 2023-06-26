library(keras)
rotate = function(x) t(apply(x, 2, rev))
imgPlot = function(img, title = ""){
  col=grey.colors(255)
  image(rotate(img), col = col, xlab = "", ylab = "", axes=FALSE, main = paste0("Label: ", as.character(title)))
}

# MNIST data set, hand written digits 0-9
## http://yann.lecun.com/exdb/mnist/
data = dataset_mnist()
train = data$train
test = data$test
oldpar = par()
par(mfrow = c(3,3))
.n = sapply(1:9, function(x) imgPlot(train$x[x,,], train$y[x]))

## normalize pixel to 0-1
train_x = array(train$x/255, c(dim(train$x), 1))
test_x = array(test$x/255, c(dim(test$x), 1))
train_y = to_categorical(train$y, 10)
test_y = to_categorical(test$y, 10)


print(dim(train_x))
print(dim(train_y))

## CNN model
model = keras_model_sequential()

model %>% 
  layer_conv_2d(input_shape = c(28L, 28L,1L),filters = 16L, kernel_size = c(2L,2L), activation = "relu") %>% 
  layer_max_pooling_2d() %>% 
  layer_conv_2d(filters = 16L, kernel_size = c(3L,3L), activation = "relu") %>% 
  layer_max_pooling_2d() %>% 
  layer_flatten() %>% 
  layer_dense(100L, activation = "relu") %>% 
  layer_dense(10L, activation = "softmax")
summary(model)

model %>% 
  compile(
    optimizer = keras::optimizer_adamax(0.01),
    loss = loss_categorical_crossentropy
  )


epochs = 10L
batch_size = 32L
model %>% 
  fit(
    x = train_x, 
    y = train_y,
    epochs = epochs,
    batch_size = batch_size,
    shuffle = TRUE,
    validation_split = 0.2
)

model %>% 
  evaluate(test_x, test_y)


## Exercise 1
## - add dropout layers
## - train/fit without fully connected layer
## - add early stopping
model = keras_model_sequential()

model %>% 
  layer_conv_2d(input_shape = c(NULL, 28, 28,1),filters = 16, kernel_size = c(2,2), activation = "relu", use_bias = F) %>% 
  layer_max_pooling_2d() %>% 
  layer_dropout(0.3) %>% 
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = "relu", use_bias = F) %>% 
  layer_max_pooling_2d() %>% 
  layer_flatten() %>% 
  layer_dense(10, activation = "softmax")
summary(model)

model %>% 
  compile(
    optimizer = keras::optimizer_adam(),
    loss = keras::loss_categorical_crossentropy,
    metrics = "accuracy"
  )

early = callback_early_stopping(patience = 5L)

epochs = 10L
batch_size = 32L
model %>% fit(
  x = train_x, 
  y = train_y,
  epochs = epochs,
  batch_size = batch_size,
  shuffle = T,
  validation_split = 0.2,
  callbacks = c(early)
)

model %>% 
  evaluate(test_x, test_y)




## Exercise 2 Cifar data set
## - fit cnn on colored images
## - keras::dataset_cifar10()

data = keras::dataset_cifar10()

train = data$train
test = data$test

image = train$x[1,,,]
image %>% 
  image_to_array() %>%
  `/`(., 255) %>%
  as.raster() %>%
  plot()

## normalize pixel to 0-1
train_x = array(train$x/255, c(dim(train$x)))
test_x = array(test$x/255, c(dim(test$x)))
train_y = to_categorical(train$y, 10)
test_y = to_categorical(test$y, 10)
model = keras_model_sequential()

model %>% 
  layer_conv_2d(input_shape = c(32L, 32L,3L),filters = 16L, kernel_size = c(2L,2L), activation = "relu") %>% 
  layer_max_pooling_2d() %>% 
  layer_dropout(0.3) %>% 
  layer_conv_2d(filters = 16L, kernel_size = c(3L,3L), activation = "relu") %>% 
  layer_max_pooling_2d() %>% 
  layer_flatten() %>% 
  layer_dense(10, activation = "softmax")
summary(model)

model %>% 
  compile(
    optimizer = optimizer_adamax(),
    loss = loss_categorical_crossentropy
  )

early = callback_early_stopping(patience = 5L)

epochs = 10L
batch_size =2L
model %>% fit(
  x = train_x, 
  y = train_y,
  epochs = epochs,
  batch_size = batch_size,
  shuffle = T,
  validation_split = 0.2,
  callbacks = c(early)
)

# settings for data augmentation
aug = image_data_generator(rotation_range = 180)

# create data generator object
train_gen = flow_images_from_data(train_x, train_y, generator = aug)

reticulate::iter_next(train_gen)[[1]][1,,,] %>% 
  image_to_array() %>%
  as.raster() %>%
  plot()



model %>% 
  fit_generator(train_gen,
                steps_per_epoch = as.integer(50000/32),
                epochs = 5L)




# Exercise -> Fit MNIST with DNN:
library(keras)
data = dataset_mnist()
train = data$train
test = data$test
x_dim = dim(train$x)
dim(train$x)
## dimension: 60000 images with 28 x 28 pixel and 1 channel
#train_x = array(train$x/255, c(60000, 28, 28, 1))

## dimension: 60000 rows with 28*28 = 784 columns
train_x = array(train$x/255, c(60000,784))
test_x = array(test$x/255, c(10000, 784))
train_y = to_categorical(train$y, 10)
test_y = to_categorical(test$y, 10)
model = keras_model_sequential()
model %>% 
  layer_dense(input_shape = c(784L), units = 100L, activation = "relu") %>% 
  layer_dense(units = 10L, activation = "softmax")
model %>% 
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adam())
model %>% 
  fit(x = train_x, y = train_y, 
      validation_data = list(test_x, test_y), epochs = 1L)

predictions = 
  model %>% 
    predict(test_x)

mean(test$y == apply(predictions, 1, which.max)-1L)
library(ranger)

rf = randomForest::randomForest(factor(y)~., data = data.frame(y = train$y, train_x)[1:1000,])

pred = predict(rf, data.frame(test_x))


mean(pred == test$y)







early = callback_early_stopping(patience = 5L)

epochs = 10L
batch_size = 32L
model %>% fit(
  x = train_x, 
  y = train_y,
  epochs = epochs,
  batch_size = batch_size,
  shuffle = T,
  validation_split = 0.2,
  callbacks = c(early)
)

model %>% 
  evaluate(test_x, test_y)



## Visualization of feature maps
K = backend()
out = K$'function'(list(model2$layers[[1]]$input, K$learning_phase()),
                   list(model2$layers[[1]]$output))
imgPlot(out(list(train_x[1,,,,drop = FALSE], 0))[[1]][1,,,1],
        which.max(train_y[1,]))





## Advanced techniques
### Data augmentation
data = dataset_mnist()
train = data$train
test = data$test
train_x = array(train$x/255, c(dim(train$x), 1))
test_x = array(test$x/255, c(dim(test$x), 1))
train_y = to_categorical(train$y, 10)
test_y = to_categorical(test$y, 10)
print(dim(train_x))
print(dim(test_y))
model = keras_model_sequential()

model %>% 
  layer_conv_2d(input_shape = c(NULL, 28, 28,1),filters = 16, kernel_size = c(2,2), activation = "relu", use_bias = F) %>% 
  layer_max_pooling_2d() %>% 
  layer_conv_2d(filters = 16, kernel_size = c(3,3), activation = "relu", use_bias = F) %>% 
  layer_max_pooling_2d() %>% 
  layer_flatten() %>% 
  layer_dense(100, activation = "relu") %>% 
  layer_dense(10, activation = "softmax")
summary(model)

model %>% 
  compile(
    optimizer = optimizer_adamax(),
    loss = loss_categorical_crossentropy
  )


aug = image_data_generator()

generator = flow_images_from_data(train_x, train_y,generator = aug)

model %>% 
  fit_generator(generator, steps_per_epoch = 10L,epochs = 5L)

# https://blogs.rstudio.com/tensorflow/posts/2017-12-14-image-classification-on-small-datasets/


### Exercise 4 




### Exercise 5 - Flower data 
data_files = list.files("flower/", full.names = TRUE)
train = data_files[str_detect(data_files, "train")]
test = readRDS(file = "flower/test.RDS")

train = lapply(train, readRDS)
train_classes = lapply(train, function(d) dim(d)[1])
train = abind::abind(train, along = 1L)

labels_train = rep(0:4, unlist(train_classes))


model = keras_model_sequential()
model %>% 
  layer_conv_2d(filters = 16L, kernel_size = c(3L,3L), input_shape = c(80L, 80L, 3L), activation = "relu") %>% 
  layer_max_pooling_2d() %>% 
  layer_flatten() %>% 
  layer_dense(units = 5L, activation = "softmax")

model %>% 
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adamax(lr = 0.001))

model %>% 
  fit(x = train, y = to_categorical(matrix(labels_train, ncol = 1L), 5L), epochs = 5L, batch_size = 32L, shuffle = TRUE)

pred = 
  model %>% 
    predict(train)
pred_classes = apply(pred, 1, which.max)

Metrics::accuracy(pred_classes-1L, labels_train)

pred_test = 
  model %>% 
    predict(test)
pred_classes = apply(pred_test, 1, which.max)
true = read.csv("Day3/flower/true.csv")

mean(pred_classes-1L == true)


### Transfer learning

data = keras::dataset_cifar10()
train = data$train
test = data$test
image = train$x[5,,,]
image %>% 
  image_to_array() %>%
  `/`(., 255) %>%
  as.raster() %>%
  plot()
train_x = array(train$x/255, c(dim(train$x)))
test_x = array(test$x/255, c(dim(test$x)))
train_y = to_categorical(train$y, 10)
test_y = to_categorical(test$y, 10)

densenet = application_densenet201(include_top = FALSE, input_shape  = c(32L, 32L, 3L))

model = keras::keras_model(inputs = densenet$input, outputs = densenet$output %>%
                              layer_flatten() %>%
                             layer_dense(units = 10L, activation = "softmax")
                             )
model %>% freeze_weights(to = length(model$layers)-1)

model %>% 
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adamax())

model %>% 
  fit(
  x = train_x, 
  y = train_y,
  epochs = 1L,
  batch_size = 32L,
  shuffle = T,
  validation_split = 0.2,
)

