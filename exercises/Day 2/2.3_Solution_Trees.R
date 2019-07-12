# Classification and Regression Trees
library(rpart)
data = airquality[complete.cases(airquality),]

rt = tree(Ozone~., data = data,control = tree.control(mincut = 30, nobs = nrow(data)))
plot(rt)
text(rt)

pred = predict(rt, data)
plot(data$Temp, data$Ozone)
lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")

sqrt(mean((data$Ozone - pred)^2))

## Exercise 1 
## - change the mincut control options. What happens?
## - compare rmse

rt = tree(Ozone~., data = data,control = tree.control(mincut = 1L, nobs = nrow(data)))
plot(rt)
text(rt)

pred = predict(rt, data)
plot(data$Temp, data$Ozone)
lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")


sqrt(mean((data$Ozone - pred)^2))



rt = tree(Ozone~., data = data,control = tree.control(mincut = 50L, nobs = nrow(data)))
plot(rt)
text(rt)

pred = predict(rt, data)
plot(data$Temp, data$Ozone)
lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")
sqrt(mean((data$Ozone - pred)^2))
### -> More splits




# Random Forest
## two random mechanisms
## 1. Bootstrap sample for each Tree
## 2. At each split, RF has to choose split variable from a random subset of variables (mtry)
## Suggest mtry: Classification sqrt(k) and Regression p/3
## nodesize -> min. number of samples in node

library(randomForest)
rf = randomForest(Ozone~., data = data)
pred = predict(rf, data)
plot(data$Temp, data$Ozone)
lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")

importance(rf)

sqrt(mean((data$Ozone - pred)^2))


## Exercise 2 
## - change mtry and nodesize 
## - compare RMSE

rf = randomForest(Ozone~., data = data, nodesize = 50L)
pred = predict(rf, data)
plot(data$Temp, data$Ozone)
lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")

importance(rf)

sqrt(mean((data$Ozone - pred)^2))


rf = randomForest(Ozone~., data = data, nodesize = 1L)
pred = predict(rf, data)
plot(data$Temp, data$Ozone)
lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")

importance(rf)

sqrt(mean((data$Ozone - pred)^2))







# Boosted Regression Trees
## in gradient boosting, the first "weak" learner is fit to the response, the following learner/trees are fit to the previous residual errors
library(xgboost)
data_xg = xgb.DMatrix(data = as.matrix(scale(data[,-1])), label = data$Ozone)
brt = xgboost(data_xg, nrounds = 500L)


par(mfrow = c(4,4))
for(i in 1:16){
  pred = predict(brt, newdata = data_xg, ntreelimit = i)
  plot(data$Temp, data$Ozone, main = i)
  lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")
}

xgboost::xgb.importance(model = brt)

sqrt(mean((data$Ozone - pred)^2))


## Exercise 3
### - change max_depth
### - try out xgboost::xgb.cv() cross validation
data_xg = xgb.DMatrix(data = as.matrix(scale(data[,-1])), label = data$Ozone)
brt = xgboost(data_xg, nrounds = 500L)

brt_cv = xgboost::xgb.cv(data = data_xg, nfold = 10L, nrounds = 10L)
print(brt_cv)





## Advanced example
x1 = seq(-3, 3, length.out = 100)
x2 = seq(-3, 3, length.out = 100)
X = expand.grid(x1, x2)
y = apply(X, 1, function(x) exp(-x[1]^2 - x[2]^2))
#y = apply(X, 1, function(x) x[1] * exp(-x[1]^2 - x[2]^2))


model = xgboost::xgboost(xgb.DMatrix(data = as.matrix(X), label = y), nrounds = 500L,verbose = 0L)
pred = predict(model, newdata = xgb.DMatrix(data = as.matrix(X)), ntreelimit = 10L)

library(animation)


saveGIF({
  for (i in c(1,2,4,8,12,20,40,80,200)) {
    pred = predict(model, newdata = xgb.DMatrix(data = as.matrix(X)), ntreelimit = i)
    image(matrix(pred, 100,100),main = paste0("Trees: ", i),axes = FALSE, las = 2)
    axis(1, at = seq(0,1, length.out = 10), labels = round(seq(-3,3, length.out = 10), 1))
    axis(2, at = seq(0,1, length.out = 10), labels = round(seq(-3,3, length.out = 10), 1), las = 2)
  }
},movie.name = "boosting.gif", autobrowse = FALSE)




## Bonus
### Implement your own BRT with rpart/tree
### See solution for an example 

#### Simulate Data
x = runif(1000,-5,5)
y = x*sin(x)*2 + rnorm(1000,0, cos(x)+1.8)
data = data.frame(x = x, y = y)
plot(y~x)

library(tree)
#### Helper function for single tree fit
get_model = function(x,y){
  control = tree.control(nobs = length(x), mincut = 20L)
  model = tree(y~x, data.frame(x = x,y = y), control = control)
  pred = predict(model, data.frame(x = x,y = y))
  return(list(model = model, pred = pred))
}

depth = 1L
pred = NULL
model_list = list()

#### Boost function
get_boosting_model = function(depth){
  m_list = list()
  
  for(i in 1:depth){
    if(i == 1) {
      m = get_model(x,y)
      pred = m$pred
    } else {
      y_res = y-pred
      m = get_model(x, y_res)
      pred = pred + m$pred
    }
    m_list[[i]] = m$model
  }
  model_list <<- m_list
  
  return(pred)
}

pred = get_boosting_model(10L)[order(data$x)]
length(model_list)
plot(model_list[[1]])


plot(y~x)
lines(x = data$x[order(data$x)],get_boosting_model(1L)[order(data$x)], col = 'red', lwd = 2)
lines(x = data$x[order(data$x)],get_boosting_model(100L)[order(data$x)], col = 'green', lwd = 2)

