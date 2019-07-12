# kknn
library(kknn)

# We want to label a new point
X = scale(iris[,1:4])
Y = iris[,5]
plot(X[-100,1], X[-100,3], col = Y)
points(X[100,1], X[100,3], col = "blue", pch = 18, cex = 1.3)
# Find the k nearest neighbors and for instance choose the label by voting
# Disadvantage, we have to calculate a distance matrix for all points
dist_X = dist(X)



# Split:
## Scaling is important with distances!
data = iris
data[,1:4] = apply(data[,1:4],2, scale)

indices = sample.int(nrow(data), 0.7*nrow(data))
train = data[indices,]
test = data[-indices,]


knn = kknn(Species~., train = train, test = test)
summary(knn)
table(test$Species, fitted(knn))

oldpar = par()
par(mfrow = c(1,2))
plot(test$Sepal.Length, test$Petal.Length, col =  predict(knn), main = "predicted")
plot(test$Sepal.Length, test$Petal.Length, col =  test$Species, main = "observed")
par(oldpar)

## Exercise 1 - airquality
## - split airquality in train and test
## - fit knn 
## - predict test values
## - visualize result (see previous exercise for trees)

data = airquality[complete.cases(airquality$Ozone) & complete.cases(airquality$Solar.R),]
data[,2:6] = apply(data[,2:6],2, scale)

indices = sample.int(nrow(data), 0.7*nrow(data))
train = data[indices,]
test = data[-indices,]

knn = kknn(Ozone~., train = train, test = test)
pred = predict(knn)
plot(test$Temp, test$Ozone)
lines(test$Temp[order(test$Temp)], pred[order(test$Temp)], col = "red")





# SVM
## SVM works by finding a hyperplane which maximes the margin, the distance between the plane and the points close to it.
library(e1071)

data = iris
data[,1:4] = apply(data[,1:4],2, scale)

indices = sample.int(nrow(data), 0.7*nrow(data))
train = data[indices,]
test = data[-indices,]


sm = svm(Species~., data = train, kernel = "linear")
pred = predict(sm, newdata = test)

oldpar = par()
par(mfrow = c(1,2))
plot(test$Sepal.Length, test$Petal.Length, col =  pred, main = "predicted")
plot(test$Sepal.Length, test$Petal.Length, col =  test$Species, main = "observed")
par(oldpar)

mean(pred==test$Species)



## Exercise - 2 airquality
## - split airquality in train and test
## - fit knn 
## - predict test values
## - visualize result (see previous exercise for trees)

data = airquality[complete.cases(airquality$Ozone) & complete.cases(airquality$Solar.R),]
data[,2:6] = apply(data[,2:6],2, scale)

indices = sample.int(nrow(data), 0.7*nrow(data))
train = data[indices,]
test = data[-indices,]

sm = svm(Ozone~., data = train)
pred = predict(sm, newdata = test)
plot(test$Temp, test$Ozone)
lines(test$Temp[order(test$Temp)], pred[order(test$Temp)], col = "red")



## Kernel-trick. SVM works fine as long as the problem is linear seperable. 
## If not we have to map the problem into a space in which it is linear seperable. That is called the kernel trick


## Advanced examples
set.seed(42)
x1 = seq(-3, 3, length.out = 100)
x2 = seq(-3, 3, length.out = 100)
X = expand.grid(x1, x2)
y = apply(X, 1, function(x) exp(-x[1]^2 - x[2]^2))
y = ifelse(1/(1+exp(-y)) < 0.62, 0, 1)



image(matrix(y, 100, 100))
saveGIF({
  for (i in c("truth","linear", "radial", "sigmoid")) {
    if(i == "truth"){
      image(matrix(y, 100,100),main = "Ground truth",axes = FALSE, las = 2)
    }else{
      sv = e1071::svm(x = X, y = factor(y), kernel = i)
      image(matrix(as.numeric(as.character(predict(sv, X))), 100,100),main = paste0("Kernel: ", i),axes = FALSE, las = 2)
      axis(1, at = seq(0,1, length.out = 10), labels = round(seq(-3,3, length.out = 10), 1))
      axis(2, at = seq(0,1, length.out = 10), labels = round(seq(-3,3, length.out = 10), 1), las = 2)
    }
  }
},movie.name = "svm.gif", autobrowse = FALSE)




## Exercise 3 - Sonar
## - split the Sonar dataset from the mlbench library into train and test
## - fit svm and knn with linear kernels, check help
## - calculate accuracies
## - fit again with different kernels and check accuracies again
library(mlbench)
data = Sonar
indices = sample.int(nrow(Sonar), 0.5*nrow(Sonar))
train = Sonar[indices,]
test = Sonar[-indices,]


sm = svm(Class~., data = train, kernel = "linear")
knn = kknn(Class~., train = train, test = test, kernel = "rectangular")

pred_sm = predict(sm, newdata = test)
pred_knn = predict(knn)

mean(pred_sm == test$Class)
mean(pred_knn == test$Class)


### change kernel:

sm = svm(Class~., data = train, kernel = "radial")
knn = kknn(Class~., train = train, test = test, kernel = "gaussian")

pred_sm = predict(sm, newdata = test)
pred_knn = predict(knn)

mean(pred_sm == test$Class)
mean(pred_knn == test$Class)
