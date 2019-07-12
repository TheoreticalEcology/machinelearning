library(keras)
library(tensorflow)
tf$enable_eager_execution()
rotate = function(x) t(apply(x, 2, rev))
imgPlot = function(img, title = ""){
  col=grey.colors(255)
  image(rotate(img), col = col, xlab = "", ylab = "", axes=FALSE, main = paste0("Label: ", as.character(title)))
}



data = dataset_mnist()
train = data$train
test = data$test

train_x = array(train$x/255, c(dim(train$x)[1], 784L))
test_x = array(test$x/255, c(dim(test$x)[1], 784L))


## Dense autoencoder
### Inputs will be compromized to two dimensions
down_size_model = keras_model_sequential()
down_size_model %>% 
  layer_dense(units = 100L, input_shape = c(784L),activation = "relu") %>% 
  layer_dense(units = 20L, activation = "relu") %>% 
  layer_dense(units = 2L, activation = "linear")

### Reconstruction of the images
up_size_model = keras_model_sequential()
up_size_model %>% 
  layer_dense(units = 20L, input_shape = c(2L), activation = "relu") %>% 
  layer_dense(units = 100L, activation = "relu") %>% 
  layer_dense(units = 784L, activation = "sigmoid")

### Combine models into one
autoencoder = keras_model(inputs = down_size_model$input, 
                          outputs = up_size_model(down_size_model$output))


autoencoder %>% 
  compile(loss = loss_binary_crossentropy, optimizer = optimizer_adamax(0.01))

image = autoencoder(train_x[1,,drop = FALSE])$numpy()
par(mfrow = c(1,2))
imgPlot(array(train_x[1,,drop = FALSE], c(28, 28)))
imgPlot(array(image, c(28, 28)))

autoencoder %>% 
  fit(x = train_x, y = train_x, epochs = 10L, batch_size = 32L)

pred_dim = down_size_model(test_x)
reconstr_pred = autoencoder(test_x)

imgPlot(array(reconstr_pred[10,]$numpy(), dim = c(28L, 28L)))
par(mfrow = c(1,1))
plot(pred_dim[,1]$numpy(), pred_dim[,2]$numpy(), col = test$y+1L)


## Generate new images!
new = matrix(c(0,0), 1, 2)
imgPlot(array(up_size_model(new)$numpy(), c(28L, 28L)))

# We can do that also with convolutional layers!
data = dataset_mnist()
train = data$train
test = data$test

train_x = array(train$x/255, c(dim(train$x), 1L))
test_x = array(test$x/255, c(dim(test$x), 1L))


down_size_model = keras_model_sequential()
down_size_model %>% 
  layer_conv_2d(filters = 16L, activation = "relu", kernel_size = c(2,2), input_shape = c(28L, 28L, 1L), strides = c(4L, 4L)) %>% 
  layer_conv_2d(filters = 8L, activation = "relu", kernel_size = c(7,7), strides = c(1L, 1L)) %>% 
  layer_flatten() %>% 
  layer_dense(units = 2L, activation = "linear")

up_size_model = keras_model_sequential()
up_size_model %>% 
  layer_dense(units = 8L, activation = "relu", input_shape = c(2L)) %>% 
  layer_reshape(target_shape = c(1L, 1L, 8L)) %>% 
  layer_conv_2d_transpose(filters = 8L, kernel_size = c(7,7), activation = "relu", strides = c(1L,1L)) %>% 
  layer_conv_2d_transpose(filters = 16L, activation = "relu", kernel_size = c(2,2), strides = c(4L,4L)) %>% 
  layer_conv_2d(filters = 1, kernel_size = c(1L, 1L), strides = c(1L, 1L), activation = "sigmoid")

autoencoder = keras_model(inputs = down_size_model$input, 
                          outputs = up_size_model(down_size_model$output))


autoencoder %>% 
  compile(loss = loss_binary_crossentropy, optimizer = optimizer_adamax(0.01))


autoencoder %>% 
  fit(x = train_x, y = train_x, epochs = 2L)

pred_dim = down_size_model(tf$constant(test_x, "float32"))
reconstr_pred = autoencoder(tf$constant(test_x, "float32"))

imgPlot(reconstr_pred[10,,,]$numpy()[,,1])

plot(pred_dim[,1]$numpy(), pred_dim[,2]$numpy(), col = test$y+1L)


## Generate new images!
new = matrix(c(10,10), 1, 2)
imgPlot(array(up_size_model(new)$numpy(), c(28L, 28L)))



## Standard autoencoder try simple to replicate the input. 
## But we want to randomly sample and create variations

# Variational autoencoder
# https://towardsdatascience.com/intuitively-understanding-variational-autoencoders-1bfe67eb5daf

data = dataset_mnist()
train = data$train
test = data$test

train_x = array(train$x/255, c(dim(train$x)[1], 784L))
test_x = array(test$x/255, c(dim(test$x)[1], 784L))


devtools::install_github("rstudio/tfprobability")


tfp = reticulate::import("tensorflow_probability")

prior = tfp$distributions$Independent(tfp$distributions$Normal(loc=tf$zeros(shape(10L,4L)), scale=1.0),
                        reinterpreted_batch_ndims=1L)

down_size_model = keras_model_sequential()
down_size_model %>% 
  layer_dense(units = 100L, input_shape = c(784L),activation = "relu") %>% 
  layer_dense(units = 20L, activation = "relu") %>% 
  layer_dense(units = 4L)

### Reconstruction of the images
up_size_model = keras_model_sequential()
up_size_model %>% 
  layer_dense(units = 20L, input_shape = c(2L), activation = "relu") %>% 
  layer_dense(units = 100L, activation = "relu") %>% 
  layer_dense(units = 784L, activation = "sigmoid")

### Combine models into one
batch_size = 32L
epochs = 10L
steps = as.integer(nrow(train_x)/32L * epochs)
prior = tfp$distributions$MultivariateNormalDiag(loc = tf$zeros(shape(batch_size, 2L), "float32"), scale_diag = tf$ones(2L, "float32"))

optimizer = tf$keras$optimizers$Adamax(0.001)
weights = c(down_size_model$weights, up_size_model$weights)

get_batch = function(){
  indices = sample.int(nrow(train_x), batch_size)
  return(train_x[indices,])
}

for(i in 1:steps){
  tmp_X = get_batch()
  with(tf$GradientTape() %as% tape, {
    encoded = down_size_model(tmp_X)
    
    dd = tfp$distributions$MultivariateNormalDiag(loc = encoded[,1:2], 
                                                  scale_diag = 1.0/(0.01+ tf$math$softplus(encoded[,3:4])))
    samples = dd$sample()
    reconstructed = up_size_model(samples)
    
    KL_loss = dd$kl_divergence(prior) # constrain
    
    loss = tf$reduce_mean(tf$negative(tfp$distributions$Binomial(1L, logits = reconstructed)$log_prob(tmp_X)))+tf$reduce_mean(KL_loss)
  })
  gradients = tape$gradient(loss, weights)
  optimizer$apply_gradients(purrr::transpose(list(gradients, weights)))
  
  if(i %% as.integer(nrow(train_x)/32L) == 0) cat("Loss: ", loss$numpy(), "\n")
}



