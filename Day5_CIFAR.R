library(keras3)

data= readRDS( "/Users/maximilianpichler/Library/CloudStorage/Dropbox/Lehre/Machine_Learning/Vorlesung/CourseMaterialPublic/ML25/cifar10.RDS")

train = data$train
test = data$test
image = train$x[1,,,]
image %>%  
  image_to_array() %>%
  `/`(., 255) %>%
  as.raster() %>%
  plot()

# prepare data
train_x = array(train$x/255, c(dim(train$x)))
test_x = array(test$x/255, c(dim(test$x)))
train_y = train$y
test_y = test$y

train_x = aperm(train_x, c(1, 4, 2, 3))
test_x = aperm(test_x, c(1, 4, 2, 3))






