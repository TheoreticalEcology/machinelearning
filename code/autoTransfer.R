Sys.setenv(CUDA_VISIBLE_DEVICES=3)
library(tensorflow)
library(keras)


data = EcoData::dataset_flower()
train = data$train
test = data$test
labels = data$labels

indices = sample.int(dim(train)[1], dim(train)[1])
train = train[indices,,,]
labels = labels[indices]

#reticulate::conda_install(packages = "keras_tuner", pip = TRUE)
kt = reticulate::import("keras_tuner")
keras = reticulate::import("keras")

build_model = function(hp) {
  pre = hp$Choice("model",values = c("B3", "B7", "dense", "X"))

  if(pre == "B3") base_model = keras$applications$EfficientNetB3(include_top = FALSE)
  if(pre == "B7") base_model = keras$applications$EfficientNetB7(include_top = FALSE)
  if(pre == "dense") base_model = keras$applications$DenseNet201(include_top = FALSE)
  if(pre == "X") base_model = keras$applications$Xception(include_top = FALSE)
  base_model = keras$applications$Xception(include_top = FALSE)
  
  if(hp$Boolean('finetune', default = FALSE)) base_model$trainable = FALSE
  inputs = keras$Input(shape = list(80L, 80L, 3L))
  x1 = base_model(inputs)
  f1 = keras$layers$Flatten()(x1)
  d1 = keras$layers$Dense(units = hp$Int("units1", 10, 200, step =1), 
                          activation = hp$Choice("act2", values = c("elu", "relu", "selu")) )(f1)
  drop = keras$layers$Dropout(rate = hp$Float("rate", 0.0, 0.5))(d1)
  output = keras$layers$Dense(units = 5L, activation = "softmax")(drop)
  
  model = keras$Model(inputs, output)
  hp_learning_rate = hp$Choice('learning_rate', values=c(1e-2, 1e-3, 1e-4, 1e-5, 1e-6))
  model$compile(optimizer=keras$optimizers$Adamax(learning_rate=hp_learning_rate,),
                loss=keras$losses$CategoricalCrossentropy,
                metrics=list('accuracy'))
  return(model)
}



tuner = kt$BayesianOptimization(build_model,
                        objective='val_loss',
                        max_trials = 300L,
                        directory='my_dir',
                        project_name='intro_to_kt')
stop_early = tf$keras$callbacks$EarlyStopping(monitor='val_loss', patience=10L)
lr = keras$callbacks$ReduceLROnPlateau(patience = 5L)

img_train = reticulate::np_array((train-125.)/125., dtype = "float32")
label_train = reticulate::np_array(keras::to_categorical(labels, num_classes = 5), dtype="float32")
tuner$search(img_train, label_train, epochs=200L, validation_split=0.2, callbacks=list(stop_early, lr)) # add learning

best_hps = tuner$get_best_hyperparameters()[[1]]
model = tuner$hypermodel$build(best_hps)
history = model$fit(img_train, label_train, epochs=250L, callbacks=list(stop_early, lr))

pred = model$predict(tf$constant((test-125.)/125., dtype = "float32"))
pred = apply(pred, 1, which.max)
write.csv(data.frame(y = pred), file = "tuner.csv")


