str(test)
summary(test)
pairs(test)
table(train$Dataset)


library(randomForest)
randomForest(Dataset ~ .,  data = train[complete.cases(train),])

fit <- glm(Dataset ~ Age + Gender + Total_Bilirubin + Alkaline_Phosphotase ,  data = train[complete.cases(train),])
summary(fit)


data <- airquality[complete.cases(airquality), ]

# create folds
data$fold <- sample.int(5, size = nrow(data), replace = T)
data$predicted = NA

# make predictions for each fold
for(i in 1:5){
  fit <- lm(Ozone ~ Temp + Solar.R, data=data[data$fold != i, ])
  data$predicted[data$fold == i] <- predict(fit, newdata = data[data$fold == i, ] )
}

MSE = sum((data$Ozone - data$predicted)^2)
RMSE = sqrt(sum((data$Ozone - data$predicted)^2))


