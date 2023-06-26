library(keras)
library(tidyverse)
data_files = list.files("flowers/", full.names = TRUE)
train = data_files[str_detect(data_files, "train")]
test = readRDS(file = "test.RDS")
train = lapply(train, readRDS)
train = abind::abind(train, along = 1L)
train = tf$concat(list(train, test), axis = 0L)$numpy()
train_x = array((train-127.5)/127.5, c(dim(train)))
get_generator = function(){
  generator = keras_model_sequential()
  generator %>% 
    layer_dense(units = 20L*20L*128L, input_shape = c(100L), use_bias = FALSE) %>% 
    layer_activation_leaky_relu() %>% 
    layer_reshape(c(20L, 20L, 128L)) %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d_transpose(filters = 256L, kernel_size = c(3L, 3L), padding = "same", strides = c(1L, 1L), use_bias = FALSE) %>% 
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d_transpose(filters = 128L, kernel_size = c(5L, 5L), padding = "same", strides = c(1L, 1L), use_bias = FALSE) %>% 
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d_transpose(filters = 64L, kernel_size = c(5L, 5L), padding = "same", strides = c(2L, 2L), use_bias = FALSE) %>%
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d_transpose(filters =3L, kernel_size = c(5L, 5L), padding = "same", strides = c(2L, 2L), activation = "tanh", use_bias = FALSE)
  return(generator)
}
generator = get_generator()
image = generator(tf$random$normal(c(1L,100L)))$numpy()[1,,,]
image = scales::rescale(image, to = c(0, 255))
image %>% 
  image_to_array() %>%
  `/`(., 255) %>%
  as.raster() %>%
  plot()
get_discriminator = function(){
  discriminator = keras_model_sequential()
  discriminator %>% 
    layer_conv_2d(filters = 64L, kernel_size = c(5L, 5L), strides = c(2L, 2L), padding = "same", input_shape = c(80L, 80L, 3L)) %>%
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d(filters = 128L, kernel_size = c(5L, 5L), strides = c(2L, 2L), padding = "same") %>% 
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_conv_2d(filters = 256L, kernel_size = c(3L, 3L), strides = c(2L, 2L), padding = "same") %>% 
    layer_activation_leaky_relu() %>% 
    layer_dropout(0.3) %>% 
    layer_flatten() %>% 
    layer_dense(units = 1L, activation = "sigmoid")
  return(discriminator)
}
discriminator = get_discriminator()
discriminator
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
epochs = 10L
steps = as.integer(nrow(train_x)/batch_size)
counter = 1
gen_loss = NULL
disc_loss = NULL
for(i in 1:(epochs*steps)){
  
  images = get_batch()
  losses = train_step(images)
  gen_loss[counter] = tf$reduce_sum(losses[[1]])$numpy()
  disc_loss[counter] = tf$reduce_sum(losses[[2]])$numpy()
  counter = counter+1
  if(i %% 10*steps == 0) {
    noise = tf$random$normal(c(1L, 100L))
    image = generator(noise)$numpy()[1,,,]
    image = scales::rescale(image, to = c(0, 255))
    image %>% 
      image_to_array() %>%
      `/`(., 255) %>%
      as.raster() %>%
      plot()
  }
  if(i %% steps == 0){
    counter = 1
    cat("Gen: ", mean(gen_loss), " Disc: ", mean(disc_loss), " \n")
  }
}

discriminator$save_weights("disc_weights2.h5")
generator$save_weights("gen_weights2.h5")
discriminator$load_weights("disc_weights2.h5")
generator$load_weights("gen_weights2.h5")

results = vector("list", 100L)
for(i in 1:100) {
  noise = tf$random$normal(c(1L, 100L))
  image = generator(noise)$numpy()[1,,,]
  image = scales::rescale(image, to = c(0, 255))
  image %>% 
    image_to_array() %>%
    `/`(., 255) %>%
    as.raster() %>%
    plot()
  results[[i]] = image
  imager::save.image(imager::as.cimg(image),quality = 1.0,file = paste0("images/flower",i, ".png"))
  imager::as.cimg(image)
}
saveRDS(abind::abind(results, along = 0L), file = "images/result.RDS")


