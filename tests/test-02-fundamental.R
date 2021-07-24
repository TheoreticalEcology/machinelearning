
testthat::test_that('##  eval=knitr::is_html_output(excludes =  epub ), results =  asis , echo = F', {testthat::expect_error( {
## ---- eval=knitr::is_html_output(excludes = "epub"), results = 'asis', echo = F---------------------------------------------------
## cat(
##   '<iframe width="560" height="315"
##   src="https://www.youtube.com/embed/X-iSQQgOd1A"
##   frameborder="0" allow="accelerometer; autoplay; encrypted-media;
##   gyroscope; picture-in-picture" allowfullscreen>
##   </iframe>'
## )


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
func = function(x) return(x^2)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

a = rnorm(100)
plot(a, func(a))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

opt = optim(1.0, func, method = "Brent", lower = -100, upper = 100)
print(opt$par)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
data = airquality[complete.cases(airquality$Ozone) & complete.cases(airquality$Solar.R),]
X = scale(data[,-1])
Y = data$Ozone


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
linear_regression = function(w){
  pred = w[1]*X[,1] + # Solar.R
         w[2]*X[,2] + # Wind
         w[3]*X[,3] + # Temp
         w[4]*X[,4] + # Month
         w[5]*X[,5] +
         w[6]         # or X * w[1:5]^T + w[6]
  # loss  = MSE, we want to find the optimal weights 
  # to minimize the sum of squared residuals.
  loss = mean((pred - Y)^2)
  return(loss)
}


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

linear_regression(runif(6))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

random_search = matrix(runif(6*5000, -10, 10), 5000, 6)
losses = apply(random_search, 1, linear_regression)
plot(losses, type = "l")
random_search[which.min(losses),]


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
opt = optim(runif(6, -1, 1), linear_regression)
opt$par


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
coef(lm(Y~X))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  echo = F', {testthat::expect_error( {
## ---- echo = F--------------------------------------------------------------------------------------------------------------------
par(mfrow = c(1,2))
curve(dexp(abs(x)), -5, 5, main = "Lasso prior")
curve(dnorm(abs(x)), -5, 5, main = "Ridge prior")


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
library(keras)
set.seed(123)

data = airquality[complete.cases(airquality),]
X = scale(data[,-1])
Y = data$Ozone
# l1/l2 on linear model.

model = keras_model_sequential()
model %>%
 layer_dense(units = 1L, activation = "linear", input_shape = list(dim(X)[2]))
summary(model)

model %>%
 compile(loss = loss_mean_squared_error, optimizer_adamax(lr = 0.5),
         metrics = c(metric_mean_squared_error))

model_history =
 model %>%
 fit(x = X, y = Y, epochs = 100L, batch_size = 20L, shuffle = TRUE)

unconstrained = model$get_weights()
summary(lm(Y~X))
coef(lm(Y~X))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(torch)
set.seed(123)

model_torch = nn_sequential(
  nn_linear(in_features = dim(X)[2], out_features = 1L)
)
opt = optim_adam(params = model_torch$parameters, lr = 0.5)

X_torch = torch_tensor(X)
Y_torch = torch_tensor(matrix(Y, ncol = 1L), dtype = torch_float32())
for(i in 1:500) {
  indices = sample.int(nrow(X), 20L)
  opt$zero_grad()
  pred = model_torch(X_torch[indices, ])
  loss = nnf_mse_loss(pred, Y_torch[indices,,drop=FALSE])
  loss$sum()$backward()
  opt$step()
}
coef(lm(Y~X))
model_torch$parameters


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
set.seed(123)

model = keras_model_sequential()
model %>% # Remind the penalty lambda that is set to 10 here.
  layer_dense(units = 1L, activation = "linear", input_shape = list(dim(X)[2]), 
              kernel_regularizer = regularizer_l1(10),
              bias_regularizer = regularizer_l1(10))
summary(model)

model %>%
  compile(loss = loss_mean_squared_error, optimizer_adamax(lr = 0.5),
          metrics = c(metric_mean_squared_error))

model_history =
  model %>%
  fit(x = X, y = Y, epochs = 30L, batch_size = 20L, shuffle = TRUE)

l1 = model$get_weights()
summary(lm(Y~X))
coef(lm(Y~X))
cbind(unlist(l1), unlist(unconstrained))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

model_torch = nn_sequential(
  nn_linear(in_features = dim(X)[2], out_features = 1L)
)
opt = optim_adam(params = model_torch$parameters, lr = 0.5)

X_torch = torch_tensor(X)
Y_torch = torch_tensor(matrix(Y, ncol = 1L), dtype = torch_float32())
for(i in 1:500) {
  indices = sample.int(nrow(X), 20L)
  opt$zero_grad()
  pred = model_torch(X_torch[indices, ])
  loss = nnf_mse_loss(pred, Y_torch[indices,,drop=FALSE])
  
  # Add l1:
  for(i in 1:length(model_torch$parameters)){
    # Remind the penalty lambda that is set to 10 here.
    loss = loss + model_torch$parameters[[i]]$abs()$sum()*10.0
  }
  
  loss$sum()$backward()
  opt$step()
}
coef(lm(Y~X))
model_torch$parameters


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  results= hide , message=FALSE, warning=FALSE', {testthat::expect_error( {
## ---- results='hide', message=FALSE, warning=FALSE--------------------------------------------------------------------------------
library(rpart)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(rpart.plot)
data = airquality[complete.cases(airquality),]


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
rt = rpart(Ozone~., data = data, control = rpart.control(minsplit = 10))
rpart.plot(rt)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
pred = predict(rt, data)
plot(data$Temp, data$Ozone)
lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(randomForest)
set.seed(123)

rf = randomForest(Ozone~., data = data)
pred = predict(rf, data)
plot(Ozone~Temp, data = data)
lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
rf$importance


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE, results= hide , message=FALSE, warning=FALSE', {testthat::expect_error( {
## ---- cache=TRUE, results='hide', message=FALSE, warning=FALSE--------------------------------------------------------------------
library(xgboost)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## BRT1, cache=TRUE', {testthat::expect_error( {
## ----BRT1, cache=TRUE-------------------------------------------------------------------------------------------------------------
set.seed(123)

data_xg = xgb.DMatrix(data = as.matrix(scale(data[,-1])), label = data$Ozone)
brt = xgboost(data_xg, nrounds = 16L, nthreads = 4L)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## BRT2, cache=TRUE', {testthat::expect_error( {
## ----BRT2, cache=TRUE-------------------------------------------------------------------------------------------------------------
par(mfrow = c(2, 2))
for(i in 1:4){
  pred = predict(brt, newdata = data_xg, ntreelimit = i)
  plot(data$Temp, data$Ozone, main = i)
  lines(data$Temp[order(data$Temp)], pred[order(data$Temp)], col = "red")
}


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## BRT3, cache=TRUE', {testthat::expect_error( {
## ----BRT3, cache=TRUE-------------------------------------------------------------------------------------------------------------
xgboost::xgb.importance(model = brt)
sqrt(mean((data$Ozone - pred)^2)) # RMSE
data_xg = xgb.DMatrix(data = as.matrix(scale(data[,-1])), label = data$Ozone)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## BRT4, cache=TRUE', {testthat::expect_error( {
## ----BRT4, cache=TRUE-------------------------------------------------------------------------------------------------------------
set.seed(123)

brt = xgboost(data_xg, nrounds = 5L)
brt_cv = xgboost::xgb.cv(data = data_xg, nfold = 3L,
                         nrounds = 3L, nthreads = 4L)
print(brt_cv)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
X = scale(iris[,1:4])
Y = iris[,5]
plot(X[-100,1], X[-100,3], col = Y)
points(X[100,1], X[100,3], col = "blue", pch = 18, cex = 1.3)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
data = iris
data[,1:4] = apply(data[,1:4],2, scale)
indices = sample.int(nrow(data), 0.7*nrow(data))
train = data[indices,]
test = data[-indices,]


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(kknn)
set.seed(123)

knn = kknn(Species~., train = train, test = test)
summary(knn)
table(test$Species, fitted(knn))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(e1071)

data = iris
data[,1:4] = apply(data[,1:4], 2, scale)
indices = sample.int(nrow(data), 0.7*nrow(data))
train = data[indices,]
test = data[-indices,]

sm = svm(Species~., data = train, kernel = "linear")
pred = predict(sm, newdata = test)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
oldpar = par(mfrow = c(1, 2))
plot(test$Sepal.Length, test$Petal.Length,
     col =  pred, main = "predicted")
plot(test$Sepal.Length, test$Petal.Length,
     col =  test$Species, main = "observed")
par(oldpar)

mean(pred == test$Species) # Accuracy.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## x1 = seq(-3, 3, length.out = 100)
## x2 = seq(-3, 3, length.out = 100)
## X = expand.grid(x1, x2)
## y = apply(X, 1, function(x) exp(-x[1]^2 - x[2]^2))
## y = ifelse(1/(1+exp(-y)) < 0.62, 0, 1)
## 
## image(matrix(y, 100, 100))
## animation::saveGIF({
##   for (i in c("truth", "linear", "radial", "sigmoid")) {
##     if(i == "truth"){
##       image(matrix(y, 100,100),
##             main = "Ground truth",axes = FALSE, las = 2)
##     }else{
##       sv = e1071::svm(x = X, y = factor(y), kernel = i)
##       image(matrix(as.numeric(as.character(predict(sv, X))), 100, 100),
##             main = paste0("Kernel: ", i),axes = FALSE, las = 2)
##       axis(1, at = seq(0,1, length.out = 10),
##            labels = round(seq(-3,3, length.out = 10), 1))
##       axis(2, at = seq(0,1, length.out = 10),
##            labels = round(seq(-3,3, length.out = 10), 1), las = 2)
##     }
##   }
## }, movie.name = "svm.gif", autobrowse = FALSE)




list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(keras)
set.seed(123)

data = airquality
summary(data)
data = data[complete.cases(data),] # Remove NAs.
summary(data)

X = scale(data[,2:6])
Y = data[,1]

model = keras_model_sequential()
penalty = 0.1
model %>%
  layer_dense(units = 100L, activation = "relu",
             input_shape = list(5L),
             kernel_regularizer = regularizer_l1(penalty)) %>%
  layer_dense(units = 100L, activation = "relu",
             kernel_regularizer = regularizer_l1(penalty) ) %>%
  layer_dense(units = 100L, activation = "relu",
             kernel_regularizer = regularizer_l1(penalty)) %>%
 # One output dimension with a linear activation function.
  layer_dense(units = 1L, activation = "linear",
             kernel_regularizer = regularizer_l1(penalty))

summary(model)

model %>%
 compile(loss = loss_mean_squared_error, keras::optimizer_adamax(0.1))

model_history =
 model %>%
 fit(x = X, y = matrix(Y, ncol = 1L), epochs = 100L,
     batch_size = 20L, shuffle = TRUE, validation_split = 0.2)

plot(model_history)
weights = lapply(model$weights, function(w) w$numpy() )
fields::image.plot(weights[[1]])


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

model_torch = nn_sequential(
  nn_linear(in_features = dim(X)[2], out_features = 100L),
  nn_relu(),
  nn_linear(100L, 100L),
  nn_relu(),
  nn_linear(100L, 100L),
  nn_relu(),
  nn_linear(100L, 1L),
)
opt = optim_adam(params = model_torch$parameters, lr = 0.1)

X_torch = torch_tensor(X)
Y_torch = torch_tensor(matrix(Y, ncol = 1L), dtype = torch_float32())
for(i in 1:500) {
  indices = sample.int(nrow(X), 20L)
  opt$zero_grad()
  pred = model_torch(X_torch[indices, ])
  loss = nnf_mse_loss(pred, Y_torch[indices,,drop=FALSE])
  
  # Add l1 (only on the 'kernel weights'):
  for(i in seq(1, 8, by = 2)){
    loss = loss + model_torch$parameters[[i]]$abs()$sum()*0.1
  }
  
  loss$sum()$backward()
  opt$step()
}


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
fields::image.plot(as.matrix(model_torch$parameters$`0.weight`))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=knitr::is_html_output(excludes =  epub ), results =  asis , echo = F', {testthat::expect_error( {
## ---- eval=knitr::is_html_output(excludes = "epub"), results = 'asis', echo = F---------------------------------------------------
## cat(
##   '<iframe width="560" height="315"
##   src="https://www.youtube.com/embed/nKW8Ndu7Mjw"
##   frameborder="0" allow="accelerometer; autoplay; encrypted-media;
##   gyroscope; picture-in-picture" allowfullscreen>
##   </iframe>'
## )


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=knitr::is_html_output(excludes =  epub ), results =  asis , echo = F', {testthat::expect_error( {
## ---- eval=knitr::is_html_output(excludes = "epub"), results = 'asis', echo = F---------------------------------------------------
## cat(
##   '<iframe width="560" height="315"
##   src="https://www.youtube.com/embed/nRtp7wSEtJA"
##   frameborder="0" allow="accelerometer; autoplay; encrypted-media;
##   gyroscope; picture-in-picture" allowfullscreen>
##   </iframe>'
## )


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  message=FALSE', {testthat::expect_error( {
## ---- message=FALSE---------------------------------------------------------------------------------------------------------------
library(keras)
library(tensorflow)
library(tidyverse)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(EcoData)
data(titanic_ml)
titanic = titanic_ml
data = titanic


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
str(data)
summary(data)
head(data)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
length(unique(data$name))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
first_split = sapply(data$name,
                     function(x) stringr::str_split(x, pattern = ",")[[1]][2])
titles = sapply(first_split,
                function(x) strsplit(x, ".",fixed = TRUE)[[1]][1])


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
table(titles)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
titles = stringr::str_trim((titles))
titles %>%
 fct_count()


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
titles2 =
  forcats::fct_collapse(titles,
                        officer = c("Capt", "Col", "Major", "Dr", "Rev"),
                        royal = c("Jonkheer", "Don", "Sir",
                                  "the Countess", "Dona", "Lady"),
                        miss = c("Miss", "Mlle"),
                        mrs = c("Mrs", "Mme", "Ms")
                        )


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
titles2 %>%  
   fct_count()


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
data =
  data %>%
    mutate(title = titles2)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
summary(data)
sum(is.na(data$age))/nrow(data)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
data =
  data %>%
    group_by(sex, pclass, title) %>%
    mutate(age2 = ifelse(is.na(age), median(age, na.rm = TRUE), age)) %>%
    mutate(fare2 = ifelse(is.na(fare), median(fare, na.rm = TRUE), fare)) %>%
    ungroup()


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
data_sub =
  data %>%
    select(survived, sex, age2, fare2, title, pclass) %>%
    mutate(age2 = scales::rescale(age2, c(0,1)),
           fare2 = scales::rescale(fare2, c(0,1))) %>%
    mutate(sex = as.integer(sex) - 1L,
           title = as.integer(title) - 1L, pclass = as.integer(pclass - 1L))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
one_title = k_one_hot(data_sub$title, length(unique(data$title)))$numpy()
colnames(one_title) = levels(data$title)

one_sex = k_one_hot(data_sub$sex, length(unique(data$sex)))$numpy()
colnames(one_sex) = levels(data$sex)

one_pclass = k_one_hot(data_sub$pclass,  length(unique(data$pclass)))$numpy()
colnames(one_pclass) = paste0(1:length(unique(data$pclass)), "pclass")


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
data_sub = cbind(data.frame(survived= data_sub$survived),
                 one_title, one_sex, age = data_sub$age2,
                 fare = data_sub$fare2, one_pclass)
head(data_sub)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
train = data_sub[!is.na(data_sub$survived),]
test = data_sub[is.na(data_sub$survived),]


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
indices = sample.int(nrow(train), 0.7*nrow(train))
sub_train = train[indices,]
sub_test = train[-indices,]


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

model = keras_model_sequential()
model %>%
  layer_dense(units = 20L, input_shape = ncol(sub_train) - 1L,
              activation = "relu") %>%
  layer_dense(units = 20L, activation = "relu") %>%
  layer_dense(units = 20L, activation = "relu") %>%
  #Output layer consists of the 1-hot encoded variable "survived" -> 2 units.
  layer_dense(units = 2L, activation = "softmax")

summary(model)

model_history =
model %>%
  compile(loss = loss_categorical_crossentropy,
          optimizer = keras::optimizer_adamax(0.01))

model_history =
  model %>%
    fit(x = as.matrix(sub_train[,-1]),
        y = to_categorical(sub_train[,1], num_classes = 2L),
        epochs = 100L, batch_size = 32L,
        validation_split = 0.2,   #Again a test set used by the algorithm.
        shuffle = TRUE)

plot(model_history)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

model_torch = nn_sequential(
  nn_linear(in_features = dim(sub_train[,-1])[2], out_features = 20L),
  nn_relu(),
  nn_linear(20L, 20L),
  nn_relu(),
  nn_linear(20L, 2L)
)
opt = optim_adam(params = model_torch$parameters, lr = 0.01)

X_torch = torch_tensor(as.matrix(sub_train[,-1])) 
Y_torch = torch_tensor(sub_train[,1]+1, dtype = torch_long())
for(i in 1:500) {
  indices = sample.int(nrow(sub_train), 20L)
  opt$zero_grad()
  pred = model_torch(X_torch[indices, ])
  loss = nnf_cross_entropy(pred, Y_torch[indices], reduction = "mean")
  print(loss)
  loss$backward()
  opt$step()
}


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
preds =
  model %>%
    predict(x = as.matrix(sub_test[,-1]))

predicted = ifelse(preds[,2] < 0.5, 0, 1) #Ternary operator.
observed = sub_test[,1]
(accuracy = mean(predicted == observed))  #(...): Show output.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
model_torch$eval()
preds_torch = nnf_softmax(model_torch(torch_tensor(as.matrix(sub_test[,-1]))),
                          dim = 2L)
preds_torch = as.matrix(preds_torch)
preds_torch = apply(preds_torch, 1, which.max)
(accuracy = mean(preds_torch-1 == observed))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
submit = 
  test %>% 
      select(-survived)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
pred = model %>% 
  predict(as.matrix(submit))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## write.csv(data.frame(y=pred[,2]), file = "Max_1.csv")


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  message=FALSE', {testthat::expect_error( {
## ---- message=FALSE---------------------------------------------------------------------------------------------------------------
library(EcoData)
library(tidyverse)
library(mlr3)
library(mlr3learners)
library(mlr3pipelines)
library(mlr3tuning)
library(mlr3measures)
data(nasa)
str(nasa)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
data = nasa %>% select(-Orbit.Determination.Date,
                       -Close.Approach.Date, -Name, -Neo.Reference.ID)
data$Hazardous = as.factor(data$Hazardous)

# Create a classification task.
task = TaskClassif$new(id = "nasa", backend = data,
                       target = "Hazardous", positive = "1")


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

# Let's create the preprocessing graph.
preprocessing = po("imputeoor") %>>% po("scale") %>>% po("encode") 

# Run the task.
transformed_task = preprocessing$train(task)[[1]]

transformed_task$missings()


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
preprocessing$plot()


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## mlr1', {testthat::expect_error( {
## ----mlr1-------------------------------------------------------------------------------------------------------------------------
set.seed(123)

transformed_task$data()
transformed_task$set_row_roles((1:nrow(data))[is.na(data$Hazardous)],
                               "validation")

cv10 = mlr3::rsmp("cv", folds = 10L)
rf = lrn("classif.ranger", predict_type = "prob")
measurement =  msr("classif.auc")


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## result = mlr3::resample(transformed_task,
##                         rf, resampling = cv10, store_models = TRUE)
## 
## # Calculate the average AUC of the holdouts.
## result$aggregate(measurement)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## mlr2, eval=FALSE', {testthat::expect_error( {
## ----mlr2, eval=FALSE-------------------------------------------------------------------------------------------------------------
## preds =
##   sapply(1:10, function(i) result$learners[[i]]$predict(transformed_task,
##                                                         row_ids = (1:nrow(data))[is.na(data$Hazardous)])$data$prob[, "1", drop = FALSE])
## dim(preds)
## predictions = apply(preds, 1, mean)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
rf$param_set


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## mlr3', {testthat::expect_error( {
## ----mlr3-------------------------------------------------------------------------------------------------------------------------
library(paradox)

rf_pars = 
    paradox::ParamSet$new(
      list(paradox::ParamInt$new("min.node.size", lower = 1, upper = 30L),
           paradox::ParamInt$new("mtry", lower = 1, upper = 30L),
           paradox::ParamLgl$new("regularization.usedepth", default = TRUE)))
print(rf_pars)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## mlr4', {testthat::expect_error( {
## ----mlr4-------------------------------------------------------------------------------------------------------------------------
set.seed(123)

inner3 = mlr3::rsmp("cv", folds = 3L)
measurement =  msr("classif.auc")
tuner =  mlr3tuning::tnr("random_search") 
terminator = mlr3tuning::trm("evals", n_evals = 5L)
rf = lrn("classif.ranger", predict_type = "prob")

learner_tuner = AutoTuner$new(learner = rf, 
                              measure = measurement, 
                              tuner = tuner, 
                              terminator = terminator,
                              search_space = rf_pars,
                              resampling = inner3)
print(learner_tuner)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## set.seed(123)
## 
## outer3 = mlr3::rsmp("cv", folds = 3L)
## result = mlr3::resample(transformed_task, learner_tuner, resampling = outer3, store_models = TRUE)
## 
## # Calculate the average AUC of the holdouts.
## result$aggregate(measurement)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## preds =
##   sapply(1:3, function(i) result$learners[[i]]$predict(transformed_task,
##                                                         row_ids = (1:nrow(data))[is.na(data$Hazardous)])$data$prob[, "1", drop = FALSE])
## dim(preds)
## predictions = apply(preds, 1, mean)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
table(data$Hazardous)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

rf_over = po("classbalancing", id = "over", adjust = "minor") %>>%rf

# However rf_over is now a "graph",
# but we can easily transform it back into a learner:
rf_over_learner = GraphLearner$new(rf_over)
print(rf_over_learner)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
rf_over_learner$param_set


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

rf_pars_over = 
    paradox::ParamSet$new(
      list(paradox::ParamInt$new("over.ratio", lower = 1, upper = 7L),
           paradox::ParamInt$new("classif.ranger.min.node.size",
                                 lower = 1, upper = 30L),
           paradox::ParamInt$new("classif.ranger.mtry", lower = 1,
                                 upper = 30L),
           paradox::ParamLgl$new("classif.ranger.regularization.usedepth",
                                 default = TRUE)))

inner3 = mlr3::rsmp("cv", folds = 3L)
measurement =  msr("classif.auc")
tuner =  mlr3tuning::tnr("random_search") 
terminator = mlr3tuning::trm("evals", n_evals = 5L)

learner_tuner_over = AutoTuner$new(learner = rf_over_learner, 
                                   measure = measurement, 
                                   tuner = tuner, 
                                   terminator = terminator,
                                   search_space = rf_pars_over,
                                   resampling = inner3)
print(learner_tuner)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## set.seed(123)
## 
## outer3 = mlr3::rsmp("cv", folds = 3L)
## result = mlr3::resample(transformed_task, learner_tuner_over
##                         resampling = outer3, store_models = TRUE)
## 
## # Calculate the average AUC of the holdouts.
## result$aggregate(measurement)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## preds =
##   sapply(1:3, function(i) result$learners[[i]]$predict(transformed_task,
##                                                         row_ids = (1:nrow(data))[is.na(data$Hazardous)])$data$prob[, "1", drop = FALSE])
## dim(preds)
## predictions = apply(preds, 1, mean)

list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
