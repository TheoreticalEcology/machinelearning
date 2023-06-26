library(keras)
library(tensorflow)
# FLower project

## Data preparation
names = c("daisy", "dandelion", "rose", "sunflower", "tulip")
labels = c(0,1,2,3,4)

files = list.files("../../../Desktop/flower/", full.names = TRUE)
files = files[stringr::str_detect(files, "train")]

data = lapply(files, readRDS)
X = abind::abind(data, along = 1L)
Y = rep(0:4, unlist(lapply(data, function(d) dim(d)[1])))

indices = sample.int(dim(X)[1], dim(X)[1])
X = X[indices,,,]
Y = Y[indices]

X = array(X/255, dim(X))

indices = sample.int(dim(X)[1], 0.8*dim(X)[1])
image = X[500,,,]
image %>% 
  image_to_array() %>%
  as.raster() %>%
  plot()


Y = to_categorical(Y, 5L)
sub_train_X = X[indices,,,]
sub_train_Y = Y[indices,]

sub_test_X = X[-indices,,,]
sub_test_Y = Y[-indices,]


## Model fitting
densenet = keras::application_densenet201(include_top = FALSE, input_shape  = c(80L, 80L, 3L))

model = keras::keras_model(inputs = densenet$input, 
                           outputs = densenet$output %>% 
                                     layer_flatten() %>% 
                                     layer_dropout(0.5) %>% 
                                     layer_dense(units = 5L, activation = "softmax")
)

#keras::freeze_weights(model, 1, length(model$layers)-1L)
summary(model)


aug = image_data_generator(rotation_range = 180, 
                           width_shift_range = 0.5,
                           height_shift_range = 0.5,
                           brightness_range = c(0.1, 0.8),
                           zoom_range = 2,
                           horizontal_flip = TRUE,
                           vertical_flip = TRUE,
                           samplewise_center = TRUE, 
                           samplewise_std_normalization = TRUE)

train_generator = 
  flow_images_from_data(sub_train_X, sub_train_Y,generator = aug, shuffle = TRUE)


aug_test = image_data_generator(samplewise_center = TRUE, 
                                samplewise_std_normalization = TRUE)
test_generator = 
  flow_images_from_data(sub_test_X, sub_test_Y, generator = aug_test)

reticulate::iter_next(train_generator)
reticulate::iter_next(test_generator)

model %>% 
  compile(loss = keras::loss_categorical_crossentropy, optimizer = keras::optimizer_adamax(0.005), metrics = c(metric_categorical_accuracy))


model %>% 
  fit_generator(train_generator, steps_per_epoch = as.integer(nrow(sub_train_X)/32),
                validation_data = test_generator, validation_steps = as.integer(nrow(sub_test_X)/32),
                epochs = 5L)

