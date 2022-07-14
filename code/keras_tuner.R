Sys.setenv(CUDA_VISIBLE_DEVICES=0)
library(tensorflow)
library(keras)

train = EcoData::dataset_flower()$train
test = EcoData::dataset_flower()$test
labels = EcoData::dataset_flower()$labels

indices = sample.int(dim(train)[1], dim(train)[1])
train = train[indices,,,]
labels = labels[indices]

# reticulate::conda_install(packages = "keras_tuner", pip = TRUE)
kt = reticulate::import("keras_tuner")
keras = reticulate::import("keras")



base_model = keras$applications$DenseNet169(include_top = FALSE)
# base_model$trainable = FALSE
inputs = keras$Input(shape = list(80L, 80L, 3L))
x1 = base_model(inputs)
d1 = keras$layers$Dense(units = 50L, activation = "relu")(x1)
output = keras$layers$Dense(units = 5L, activation = "softmax")(d1)
model = keras$Model(inputs, output)
summary(model)
keras$callbacks$ReduceLROnPlateau()
model_builder = function(hp) {
  model = keras$Sequential()
  model$add(keras$layers$Conv2D(input_shape = c(80L, 80L, 3L),
                                filters = hp$Int('filters1', min_value=4L, max_value=96L, step=4L), 
                                kernel_size = hp$Int('kernelS1', min_value=2L, max_value=10L, step=1L),
                                activation = hp$Choice("act1", values = c("elu", "relu", "selu"))
  ) )
  if(hp$Boolean("b1", default=TRUE)) model$add(keras$layers$BatchNormalization())
  model$add(keras$layers$Dropout(rate = hp$Float('rate1', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Conv2D(filters = hp$Int('filters2', min_value=4L, max_value=96L, step=4L), 
                                kernel_size = hp$Int('kernelS2', min_value=2L, max_value=10L, step=1L),
                                activation = hp$Choice("act2", values = c("elu", "relu", "selu"))
                                ) )
  if(hp$Boolean("b2", default=TRUE)) model$add(keras$layers$BatchNormalization())
  model$add(keras$layers$Dropout(rate = hp$Float('rate3', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Conv2D(filters = hp$Int('filters3', min_value=4L, max_value=96L, step=4L), 
                                kernel_size = hp$Int('kernelS3', min_value=2L, max_value=10L, step=1L),
                                activation = hp$Choice("act3", values = c("elu", "relu", "selu"))
  ) )
  if(hp$Boolean("b3", default=TRUE)) model$add(keras$layers$BatchNormalization())
  model$add(keras$layers$Dropout(rate = hp$Float('rate3', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Conv2D(filters = hp$Int('filters4', min_value=4L, max_value=96L, step=4L), 
                                kernel_size = hp$Int('kernelS4', min_value=2L, max_value=10L, step=1L),
                                activation = hp$Choice("act4", values = c("elu", "relu", "selu"))
  ) )
  if(hp$Boolean("b4", default=TRUE)) model$add(keras$layers$BatchNormalization())
  model$add(keras$layers$Dropout(rate = hp$Float('rate4', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Flatten())
  model$add(keras$layers$Dense(units = hp$Int('units', min_value=20L, max_value=100L, step=1L)))
  model$add(keras$layers$Dropout(rate = hp$Float('rate5', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Dense(units = 5L, activation = "softmax"))
  


# Tune the learning rate for the optimizer
# Choose an optimal value from 0.01, 0.001, or 0.0001
hp_learning_rate = hp$Choice('learning_rate', values=c(1e-2, 1e-3, 1e-4))

model$compile(optimizer=keras$optimizers$Adamax(learning_rate=hp_learning_rate),
              loss=keras$losses$CategoricalCrossentropy(),
              metrics=list('accuracy'))

return(model)
}
tuner = kt$Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=80L,
                     factor=3L,
                     directory='my_dir',
                     project_name='intro_to_kt')
stop_early = tf$keras$callbacks$EarlyStopping(monitor='val_loss', patience=8L)
img_train = reticulate::np_array(train/255., dtype = "float32")
label_train = reticulate::np_array(keras::to_categorical(labels, num_classes = 5), dtype="float32")
tuner$search(img_train, label_train, epochs=80L, validation_split=0.2, callbacks=list(stop_early))

best_hps = tuner$get_best_hyperparameters()[[1]]
model = tuner$hypermodel$build(best_hps)
history = model$fit(img_train, label_train, epochs=50L, validation_split=0.2)






model_builder = function(hp) {
  model = keras$Sequential()
  model$add(keras$layers$Conv2D(input_shape = c(80L, 80L, 3L),
                                filters = hp$Int('filters1', min_value=4L, max_value=96L, step=4L), 
                                kernel_size = hp$Int('kernelS1', min_value=2L, max_value=10L, step=1L),
                                activation = hp$Choice("act1", values = c("elu", "relu", "selu"))
  ) )
  if(hp$Boolean("b1", default=TRUE)) model$add(keras$layers$BatchNormalization())
  model$add(keras$layers$Dropout(rate = hp$Float('rate1', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Conv2D(filters = hp$Int('filters2', min_value=4L, max_value=96L, step=4L), 
                                kernel_size = hp$Int('kernelS2', min_value=2L, max_value=10L, step=1L),
                                activation = hp$Choice("act2", values = c("elu", "relu", "selu"))
  ) )
  if(hp$Boolean("b2", default=TRUE)) model$add(keras$layers$BatchNormalization())
  model$add(keras$layers$Dropout(rate = hp$Float('rate3', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Conv2D(filters = hp$Int('filters3', min_value=4L, max_value=96L, step=4L), 
                                kernel_size = hp$Int('kernelS3', min_value=2L, max_value=10L, step=1L),
                                activation = hp$Choice("act3", values = c("elu", "relu", "selu"))
  ) )
  if(hp$Boolean("b3", default=TRUE)) model$add(keras$layers$BatchNormalization())
  model$add(keras$layers$Dropout(rate = hp$Float('rate3', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Conv2D(filters = hp$Int('filters4', min_value=4L, max_value=96L, step=4L), 
                                kernel_size = hp$Int('kernelS4', min_value=2L, max_value=10L, step=1L),
                                activation = hp$Choice("act4", values = c("elu", "relu", "selu"))
  ) )
  if(hp$Boolean("b4", default=TRUE)) model$add(keras$layers$BatchNormalization())
  model$add(keras$layers$Dropout(rate = hp$Float('rate4', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Flatten())
  model$add(keras$layers$Dense(units = hp$Int('units', min_value=20L, max_value=100L, step=1L)))
  model$add(keras$layers$Dropout(rate = hp$Float('rate5', min_value=0.0, max_value=0.3)))
  model$add(keras$layers$Dense(units = 5L, activation = "softmax"))
  
  
  
  # Tune the learning rate for the optimizer
  # Choose an optimal value from 0.01, 0.001, or 0.0001
  hp_learning_rate = hp$Choice('learning_rate', values=c(1e-2, 1e-3, 1e-4))
  
  model$compile(optimizer=keras$optimizers$Adamax(learning_rate=hp_learning_rate),
                loss=keras$losses$CategoricalCrossentropy(),
                metrics=list('accuracy'))
  
  return(model)
}
tuner = kt$Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=80L,
                     factor=3L,
                     directory='my_dir',
                     project_name='intro_to_kt')
stop_early = tf$keras$callbacks$EarlyStopping(monitor='val_loss', patience=8L)
img_train = reticulate::np_array(train/255., dtype = "float32")
label_train = reticulate::np_array(keras::to_categorical(labels, num_classes = 5), dtype="float32")
tuner$search(img_train, label_train, epochs=80L, validation_split=0.2, callbacks=list(stop_early))

best_hps = tuner$get_best_hyperparameters()[[1]]
model = tuner$hypermodel$build(best_hps)
history = model$fit(img_train, label_train, epochs=50L, validation_split=0.2)







train = EcoData::dataset_flower()$train/255.
test = EcoData::dataset_flower()$test/255.
labels = EcoData::dataset_flower()$labels

indices = sample.int(dim(train)[1], dim(train)[1])
train = train[indices,,,]
labels = labels[indices]

densenet = keras::application_efficientnet_b5(include_top = FALSE,
                                          input_shape = list(80L, 80L, 3L))

keras::freeze_weights(densenet)

model = keras_model(inputs = densenet$input, 
                    outputs = densenet$output %>%
                      layer_flatten() %>%
                      # layer_dropout(0.2) %>%
                      # layer_dense(units = 50) %>%
                      # layer_dropout(0.2) %>%
                      layer_dense(units = 5L, activation = "softmax"))

# Data augmentation.
aug = image_data_generator(rotation_range = 90, zoom_range = 0.4,
                           width_shift_range = 0.2, height_shift_range = 0.2,
                           vertical_flip = TRUE, horizontal_flip = TRUE, validation_split = 0.2)

# Data preparation / splitting.
indices = sample.int(nrow(train), 0.1 * nrow(train))

generator = flow_images_from_data(train[-indices,,,],
                                  k_one_hot(labels[-indices], num_classes = 5L), 
                                  batch_size = 25L, shuffle = TRUE,
                                  generator = aug)


## Training loop with early stopping:

# As we use an iterator (the generator), validation loss is not applicable.
# An available metric is the normal loss.
early = keras::callback_early_stopping(patience = 5L, monitor = "loss")

model %>%
  keras::compile(loss = loss_categorical_crossentropy,
                 optimizer = keras::optimizer_rmsprop(learning_rate = 0.002), metrics = list("accuracy"))

model %>%
  fit(generator, epochs = 80L, batch_size = 45L,
      shuffle = TRUE, callbacks = c(early), 
      validation_data = list(x = train[indices,,,], y = k_one_hot(labels[indices], num_classes = 5L)) )

