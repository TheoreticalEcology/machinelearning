library(tensorflow)
exists("tf")
tf$enable_eager_execution()


# TensorFlow data structures
a = tf$constant(5)
b = tf$constant(10)
c = tf$add(a, b)

## tensorflow:::`+.tensorflow.tensor`()
d = c + 5
class(d)

## convert to R:
d$numpy()


##
r_matrix = matrix(runif(10*10), 10,10)
m = tf$constant(r_matrix)
m / 2


## Data types - good practise with R-TF
r_matrix = matrix(runif(10*10), 10,10)
m = tf$constant(r_matrix, dtype = "float64")
b = tf$constant(2.0, dtype = "float64")

c = m / b

## tensorflow arguments require also exact/explicit data types:
matrix(t(r_matrix), 5, 20, byrow = TRUE)
tf$reshape(r_matrix, shape = c(5L, 20L))$numpy()

## exploring tf, tf$math and tf$linalg
set.seed(42)
x = rnorm(100)
max(x)
tf$math$reduce_max(x)

min(x)
tf$math$reduce_min(x)

mean(x)
tf$math$reduce_mean(x)

which.max(x)
tf$argmax(x)

which.min(x)
tf$argmin(x)

order(x)
tf$argsort(x)

m = matrix(runif(9), 3, 3)
solve(m)
tf$linalg$inv(m)

diag(m)
tf$linalg$diag_part(m)

diag(diag(m))
tf$linalg$diag(tf$linalg$diag_part(m))

eigen(m)
tf$linalg$eigh(m)

det(m)
tf$linalg$det(m)
