# Datentypen
# devtools::install_github('araastat/reprtree')


iris

str(iris)

plot(iris)

library(randomForest)

# 1. Regression - Sepal.length ~ Pental.Length
library(randomForest)

m1 <- randomForest(Sepal.Length ~ ., data = iris)
m1
str(m1)
m1$type

predict(m1)

par(mfrow = c(1,2))

reprtree:::plot.getTree(m1, iris)

plot(predict(m1), iris$Sepal.Length, xlab = "predicted", ylab = "observed")
abline(0,1)

varImpPlot(m1)

# 2. Classification - Species ~ .

set.seed(123)

m1 <- randomForest(Species ~ ., data = iris)
m1
str(m1)
m1$type

predict(m1)

par(mfrow = c(1,2))

reprtree:::plot.getTree(m1, iris)

varImpPlot(m1)

oldpar <- par(mfrow = c(1,2))
plot(iris$Petal.Width, iris$Petal.Length, col = iris$Species, main = "observed")
plot(iris$Petal.Width, iris$Petal.Length, col = predict(m1), main = "predicted")
par(oldpar)

# confusion matrix
table(predict(m1),iris$Species)

set.seed(123)

sIris = scale(iris[,1:4])
result<- kmeans(sIris,3) #aplly k-means algorithm with no. of centroids(k)=3
result$size # gives no. of records in each cluster

result$centers # gives value of cluster center datapoint value(3 centers for k=3)

result$cluster #gives cluster vector showing the custer where each record falls


# 3. Unsupervised - Species unknown:

sIris = scale(iris[,1:4])
model<- kmeans(sIris,3) # aplly k-means algorithm with no. of centroids(k)=3
par(mfrow = c(1,2))
plot(Petal.Length~Petal.Width, data = sIris, col = model$cluster, main = "Predicted clusters")
plot(Petal.Length~Petal.Width, data = sIris, col = iris$Species, main = "True species")

# confusion matrix
table(model$cluster,iris$Species)

