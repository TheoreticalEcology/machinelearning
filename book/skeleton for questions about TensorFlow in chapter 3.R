library(tensorflow)
library(keras)


x = as.double(1:100)
y = 1:100

max(x) # R solution. Integer!
tf$math$reduce_max(x) # TensorFlow solution. Integer!

max(y)  # Float!
tf$math$reduce_max(y) # Float!

tf$math$reduce_min(x) # Integer!
tf$math$reduce_min(y) # Float!

# Check out the difference here:
mean(x)
mean(y)
tf$math$reduce_mean(x)  # Integer!
tf$math$reduce_mean(y)  # Float!

