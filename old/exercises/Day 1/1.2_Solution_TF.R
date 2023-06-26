# Introduction to tensorflow
## **** Objective of this lesson: familiarize yourself with TF and its data structures ****

## Background:
## TF is a math library which is highly optimized for neural networks
## If a GPU is available, computations can be easily run on the GPU but even on the CPU is TF still very fast
## The "backend" (i.e. all the functions and all computations) are written in C++ and CUDA (CUDA is a programming language for the GPU)
## The interface (the part of TF that we use) is written in python and is also available in R, which means, we can write the code in R/Python but it will be executed by the (compiled) C++ backend. 

library(tensorflow)
# Don't worry about weird messages. TF supports additional optimizations
exists("tf")


# TensorFlow data structures
## TF has two data containers (structures):
##  - constant (tf$constant) --> creates a constant (immutable) value in the computation graph
##  - variable (tf$Variable) --> creates a mutable value in the computation graph (used as parameter/weight in models)
a = tf$constant(5)
b = tf$constant(10)
print(a)
print(b)
c = tf$add(a, b)
print(c)
tf$print(c)

## the tensorflow library (created by the RStudio team) built R methods for all common operations:
`+.tensorflow.tensor` = function(a, b) return(tf$add(a,b))
tf$print(a+b)

## their operators also transfrom automatically R numbers into constant tensors when attempting to add a tensor to a R number:
d = c + 5  # 5 is automatically converted to a tensor
print(d)

## TF container are objects, which means that they are not just simple variables of type numeric (class(5)), and they have methods
## For instance, there is a method to transform the tensor object back to a R object:
## convert to R:
class(d)
class(d$numpy())


## Data types - good practise with R-TF
## R uses dynamic typing, which means you can assign to a variable a number, character, function or whatever, 
## and you do not have to tell R the type, R infers the type automatically, 
## in other languages you have to state explicitly the type, e.g. C: int a = 5; float a = 5.0; char a = "a";
## While TF tries to infer dynamically the type, often you must state it explicitly.
## Common important types: float32 (floating point number with 32bits, "single precision"), float64 ("double precision"), int8 (integer with 8bits)
## Why does TF support float32 numbers when most cpus today can handle easily 64bit numbers? 
## --> Many GPUs (e.g. the NVIDIA geforces) can handle only up to 32bit numbers! (you do not need high precision in graphical modeling)
r_matrix = matrix(runif(10*10), 10,10)
m = tf$constant(r_matrix, dtype = "float64") # isntead of the string, you can also provde the tf$float64 object
b = tf$constant(2.0, dtype = "float64")
c = m / b

m = tf$constant(r_matrix, dtype = tf$float32) 
b = tf$constant(2.0, dtype = tf$float64)
c = m / b # doesn't work! we try to divide float32/float64


## tensorflow arguments often require also exact/explicit data types:
## TF often expects for arguments integer, however, in R we have need a "L" after an integer to tell the R interpreter that it should be treated as an integer
is.integer(5)
is.integer(5L)
matrix(t(r_matrix), 5, 20, byrow = TRUE)
tf$reshape(r_matrix, shape = c(5, 20))$numpy()
tf$reshape(r_matrix, shape = c(5L, 20L))$numpy()


# Exercises 
## You can access the different mathematical operations in tf via tf$..., e.g. there is a tf$math$... for all common math operations
## or the tf$linalg$... for different linear algebra operations 
## Tip: type tf$ and then hit the tab key to list all available options (often you have to do this directly in the console)

## exploring tf, tf$math and tf$linalg
## Transfer/translate/re-do all R operations in tf operations:
## Tip: use google, e.g. type into google "tensorflow calculate sum" 
##     (the tf documentation is written for python users, just exchange the point with a dollar: tf.math.log --> tf$math$log)

set.seed(42)
x = rnorm(100)
## An example:
max(x)
# Solution:
tf$math$reduce_max(x)

# a)
min(x)
tf$math$reduce_min(x)

# b)
mean(x)
tf$math$reduce_mean(x)

# c)
which.max(x) # use google!
tf$argmax(x) 

# d)
which.min(x)
tf$argmin(x)

# e)
order(x) # use google!
tf$argsort(x)

# f)
m = matrix(x, 10, 10) # see tf$reshape
m_2 = abs(  m %*% m  )
m_2_log = log(m_2)


mt = tf$reshape(x, list(10L, 10L))
m_2t = tf$math$abs( tf$matmul(mt, tf$transpose(mt)) )
m_2_logt = tf$math$log(m_2t)

# g) custom mean function i.e. rewrite the function using tensorflow 
mean_r = function(x) {
  result = sum(x) / length(x)
  return(result)
}
mean_r(x) == mean(x)

mean_tf = function(x) {
  result = tf$math$reduce_sum(x)
  return( result / length(x) )  # if x is a R object 
}


# h) comparison of R to TF speed:
## Tip: check out the tf.reduce_mean documentation and the "axis" argument
## Anmerkungen Johannes: nirgendwo erklaert, dass ich die funktion in tf schreiben soll 
## ich würde mean_per_row statt max_per_row als name verwenden einfach um 
## keine Unklarheiten aufzuwerfen 

mean_per_row_r = function(x = matrix(0.0, 10L, 10L)) {
  max_per_row = apply(x, 1, mean)
  result = m2 - max_per_row
  return(result)
}

mean_per_row_tf = function(x = matrix(0.0, 10L, 10L)) {
  x = tf$constant(x)
  max_per_row = tf$reduce_mean(x,  axis = 0L)
  result = x - max_per_row
  return(result)
}

## Try different matrix sizes!
test = matrix(0.0, 100L, 100L)
microbenchmark::microbenchmark(mean_per_row_r(test), mean_per_row_tf(test))
## Why is R always faster?
## a) the R functions we used (apply, mean, "-") are also implemented in C
## b) the Problem is not large enough and TF has an overhead

## Anmerkungen Johannes: Könnten wir hier ein Beispiel dabei haben wo TF schneller ist, 
## einfach aus pädagogischen Gründen um den Leuten den Sinn von TF zu erklären
## 2. Wissen die Leute was die ganzen Dinge machen? Denke wir könnten hier einen Link 
## zu ner Erklärung hochstellen und sagen: Hey ML brauchst du vernünftige Kenntnisse in 
## Linearer Algebra 


# Bonus exercises:
# i)
m = matrix(runif(9), 3, 3)
solve(m)
tf$linalg$inv(m)

# j)
diag(m)
tf$linalg$diag_part(m)

# k)
diag(diag(m))
tf$linalg$diag(tf$linalg$diag_part(m))

# l)
eigen(m)
tf$linalg$eigh(m)

# m)
det(m)
tf$linalg$det(m)
