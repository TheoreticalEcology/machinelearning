library(cito)

rotate = function(x){ t(apply(x, 2, rev)) }

imgPlot = function(img, title = ""){
  col = grey.colors(255)
  image(rotate(img), col = col, xlab = "", ylab = "", axes = FALSE,
        main = paste0("Label: ", as.character(title)))
}


data = readRDS("/Users/maximilianpichler/Library/CloudStorage/Dropbox/Lehre/Machine_Learning/Vorlesung/CourseMaterialPublic/ML25/mnist_data.RDS")

train = data$train
test = data$test
dim(train$x)

# normalize pixels to [0, 1]
train_x = array(train$x, c(60000, 1, 28, 28))
max(train_x)
min(train_x)
train_x = train_x/255
max(train_x)

labels = train$y + 1

# build model
library(cito)
architecture = create_architecture(
  cito::conv(n_kernels = 16L, kernel_size = 2, activation = "relu"),
  maxPool(2L),
  conv(n_kernels = 16L, kernel_size = 3L, activation = "relu"),
  maxPool(),
  linear(50L, activation = "relu")
)

# fit model
model.cito = cnn(X = train_x[1:1000,,,,drop=F], Y = labels[1:1000],
                 loss = "softmax", # multiclass
                 architecture = architecture,
                 batchsize = 200L,
                 epochs = 200L,
                 optimizer = "adam",
                 validation = 0.2,
                 )

# predictions
test_x = array(test$x/255, c(10000, 1, 28, 28))
pred = predict(model.cito, test_x, type="response")
pred[1:5,]
pred_labels = apply(pred, 1, which.max)

mean((pred_labels - 1) == test$y)


plot(model.cito)













