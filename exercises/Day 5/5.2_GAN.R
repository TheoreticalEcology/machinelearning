library(keras)
library(tensorflow)
rotate = function(x) t(apply(x, 2, rev))
imgPlot = function(img, title = ""){
  col=grey.colors(255)
  image(rotate(img), col = col, xlab = "", ylab = "", axes=FALSE, main = paste0("Label: ", as.character(title)))
}


# GAN with DNN:

data = dataset_mnist()
train = data$train
test = data$test

train_x = array((train$x-127.5)/127.5, c(dim(train$x)[1], 784L))
test_x = array((test$x-127.5)/127.5, c(dim(test$x)[1], 784L))

get_generator = function(){
  generator = keras_model_sequential()
  generator %>% 
    layer_dense(units = 200L ,input_shape = c(100L)) %>% 
    layer_activation_leaky_relu() %>% 
    layer_dense(units = 200L) %>% 
    layer_activation_leaky_relu() %>% 
    layer_dense(units = 784L, activation = "tanh")
  return(generator)
}
generator = get_generator()
sample = tf$random$normal(c(1L, 100L))
imgPlot(array(generator(sample)$numpy(), c(28L, 28L)))

get_discriminator = function(){
  discriminator = keras_model_sequential()
  discriminator %>% 
    layer_dense(units = 200L, input_shape = c(784L)) %>% 
    layer_activation_leaky_relu() %>% 
    layer_dense(units = 100L) %>% 
    layer_activation_leaky_relu() %>% 
    layer_dense(units = 1L, activation = "sigmoid")
  return(discriminator)
}

discriminator = get_discriminator()
discriminator(generator(tf$random$normal(c(1L, 100L))))

ce = tf$keras$losses$BinaryCrossentropy(from_logits = TRUE)
loss_discriminator = function(real, fake){
  real_loss = ce(tf$ones_like(real), real)
  fake_loss = ce(tf$zeros_like(fake), fake)
  return(real_loss+fake_loss)
}
loss_generator = function(fake){
  return(ce(tf$ones_like(fake), fake))
}

gen_opt = tf$keras$optimizers$RMSprop(1e-4)
disc_opt = tf$keras$optimizers$RMSprop(1e-4)
batch_size = 32L
get_batch = function(){
  indices = sample.int(nrow(train_x), batch_size)
  return(tf$constant(train_x[indices,], "float32"))
}

train_step = function(images){
  noise = tf$random$normal(c(32L, 100L))
  with(tf$GradientTape(persistent = TRUE) %as% tape,{
    gen_images = generator(noise)
    fake_output = discriminator(gen_images)
    real_output = discriminator(images)
    gen_loss = loss_generator(fake_output)
    disc_loss = loss_discriminator(real_output, fake_output)
  })
  gen_grads = tape$gradient(gen_loss, generator$weights)
  disc_grads = tape$gradient(disc_loss, discriminator$weights)
  rm(tape)
  gen_opt$apply_gradients(purrr::transpose(list(gen_grads, generator$weights)))
  disc_opt$apply_gradients(purrr::transpose(list(disc_grads, discriminator$weights)))
  return(c(gen_loss, disc_loss))
}
train_step = tf$`function`(reticulate::py_func(train_step))


steps = as.integer(nrow(train_x)/batch_size)
generator = get_generator()
discriminator = get_discriminator()
epochs = 30L
steps = as.integer(nrow(train_x)/batch_size)
counter = 1
gen_loss = NULL
disc_loss = NULL
for(i in 1:(epochs*steps)){
  images = get_batch()
  losses = train_step(images)
  gen_loss = tf$reduce_sum(losses[[1]])$numpy()
  disc_loss = tf$reduce_sum(losses[[2]])$numpy()
  if(i %% 50*steps == 0) {
    noise = tf$random$normal(c(1L, 100L))
    imgPlot(array(generator(noise)$numpy(), c(28L, 28L)), "Gen")
  }
  if(i %% steps == 0){
    counter = 1
    cat("Gen: ", mean(gen_loss), " Disc: ", mean(disc_loss), " \n")
  }
}














# GAN with CNN

data = dataset_mnist()
train = data$train
test = data$test
train_x = array((train$x-127.5)/127.5, c(dim(train$x), 1L))
get_generator = function(){
  generator = keras_model_sequential()
  generator %>% 
    layer_dense(units = 7*7*256, input_shape = c(100L), use_bias = FALSE) %>% 
    layer_activation_leaky_relu() %>% 
    layer_reshape(c(7L, 7L, 256L)) %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d_transpose(filters = 128L, kernel_size = c(5L, 5L), padding = "same", strides = c(1L, 1L), use_bias = FALSE) %>% 
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d_transpose(filters = 64L, kernel_size = c(5L, 5L), padding = "same", strides = c(2L, 2L), use_bias = FALSE) %>%
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d_transpose(filters = 1L, kernel_size = c(5L, 5L), padding = "same", strides = c(2L, 2L), activation = "tanh", use_bias = FALSE)
  return(generator)
}
generator = get_generator()
imgPlot(generator(tf$random$normal(c(1L, 100L)))$numpy()[1,,,1])
get_discriminator = function(){
  discriminator = keras_model_sequential()
  discriminator %>% 
    layer_conv_2d(filters = 64L, kernel_size = c(5L, 5L), strides = c(2L, 2L), padding = "same", input_shape = c(28L, 28L, 1L)) %>%
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d(filters = 128L, kernel_size = c(5L, 5L), strides = c(2L, 2L), padding = "same") %>% 
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_flatten() %>% 
    layer_dense(units = 1L, activation = "sigmoid")
  return(discriminator)
}

discriminator = get_discriminator()
discriminator(generator(tf$random$normal(c(1L, 100L))))
ce = tf$keras$losses$BinaryCrossentropy(from_logits = TRUE,label_smoothing = 0.1)
loss_discriminator = function(real, fake){
  real_loss = ce(tf$ones_like(real), real)
  fake_loss = ce(tf$zeros_like(fake), fake)
  return(real_loss+fake_loss)
}
loss_generator = function(fake){
  return(ce(tf$ones_like(fake), fake))
}
gen_opt = tf$keras$optimizers$RMSprop(1e-4)
disc_opt = tf$keras$optimizers$RMSprop(1e-4)
batch_size = 32L
get_batch = function(){
  indices = sample.int(nrow(train_x), batch_size)
  return(tf$constant(train_x[indices,,,,drop=FALSE], "float32"))
}
train_step = function(images){
  noise = tf$random$normal(c(32L, 100L))
  
  with(tf$GradientTape(persistent = TRUE) %as% tape,{
    gen_images = generator(noise)
    
    real_output = discriminator(images)
    fake_output = discriminator(gen_images)
    
    gen_loss = loss_generator(fake_output)
    disc_loss = loss_discriminator(real_output, fake_output)
    
  })
  
  gen_grads = tape$gradient(gen_loss, generator$weights)
  disc_grads = tape$gradient(disc_loss, discriminator$weights)
  rm(tape)
  
  gen_opt$apply_gradients(purrr::transpose(list(gen_grads, generator$weights)))
  disc_opt$apply_gradients(purrr::transpose(list(disc_grads, discriminator$weights)))
  
  return(c(gen_loss, disc_loss))
  
}
train_step = tf$`function`(reticulate::py_func(train_step))
steps = as.integer(nrow(train_x)/batch_size)
generator = get_generator()
discriminator = get_discriminator()
epochs = 30L
steps = as.integer(nrow(train_x)/batch_size)
counter = 1
gen_loss = NULL
disc_loss = NULL
for(i in 1:(epochs*steps)){
  
  images = get_batch()
  losses = train_step(images)
  gen_loss = tf$reduce_sum(losses[[1]])$numpy()
  disc_loss = tf$reduce_sum(losses[[2]])$numpy()
  
  if(i %% 10*steps == 0) {
    noise = tf$random$normal(c(1L, 100L))
    imgPlot(generator(noise)$numpy(), "Gen")
  }
  if(i %% steps == 0){
    counter = 1
    cat("Gen: ", mean(gen_loss), " Disc: ", mean(disc_loss), " \n")
  }
}





## noisy labels
## label switching

generator = get_generator()
discriminator = get_discriminator()

loss_discriminator = function(real, fake){
  ones = tf$random$uniform(real$shape, minval = 0.9,maxval = 1.0)
  zeros = tf$random$uniform(fake$shape, minval = 0.0, maxval = 0.1)
  real_loss = ce(zeros, real)
  fake_loss = ce(ones, fake)
  return(real_loss+fake_loss)
}

loss_generator = function(fake){
  zeros = tf$random$uniform(fake$shape, minval = 0.0,maxval =  0.1)
  
  return(ce(zeros, fake))
}

gen_opt = tf$keras$optimizers$RMSprop(1e-5)
disc_opt = tf$keras$optimizers$RMSprop(1e-5)

train_step = function(images){
  noise = tf$random$normal(c(32L, 100L))
  
  with(tf$GradientTape(persistent = TRUE) %as% tape,{
    gen_images = generator(noise)
    
    real_output = discriminator(images)
    fake_output = discriminator(gen_images)
    
    gen_loss = loss_generator(fake_output)
    disc_loss = loss_discriminator(real_output, fake_output)
    
  })
  
  gen_grads = tape$gradient(gen_loss, generator$weights)
  disc_grads = tape$gradient(disc_loss, discriminator$weights)
  rm(tape)
  
  gen_opt$apply_gradients(purrr::transpose(list(gen_grads, generator$weights)))
  disc_opt$apply_gradients(purrr::transpose(list(disc_grads, discriminator$weights)))
  
  return(c(gen_loss, disc_loss))
  
}
train_step = tf$`function`(reticulate::py_func(train_step))

epochs = 1L
steps = as.integer(nrow(train_x)/batch_size)

for(i in 1:(epochs*steps)){
  
  images = get_batch()
  losses = train_step(images)
  gen_loss = tf$reduce_sum(losses[[1]])$numpy()
  disc_loss = tf$reduce_sum(losses[[2]])$numpy()
  
  if(i %% 10*steps == 0) {
    noise = tf$random$normal(c(1L, 100L))
    cat("Gen: ", gen_loss, " Disc: ", disc_loss, " \n")
    imgPlot(generator(noise)$numpy(), "Gen")
  }
}

