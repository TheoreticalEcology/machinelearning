Sys.setenv(CUDA_VISIBLE_DEVICES=3)
library(tensorflow)
library(keras)

train = EcoData::dataset_flower()$train
test = EcoData::dataset_flower()$test
labels = EcoData::dataset_flower()$labels

indices = sample.int(dim(train)[1], dim(train)[1])
train = train[indices,,,]
labels = labels[indices]

ak = reticulate::import("autokeras")
clf = ak$ImageClassifier(num_classes = 5L, max_trials = 1L, objective = "val_accuracy")
clf$fit(x = reticulate::np_array(train), y = reticulate::np_array(labels))
results = clf$predict(reticulate::np_array(test))
saveRDS(results, "predictions.RDS")