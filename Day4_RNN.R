utils::download.file("https://www.dropbox.com/s/radyscnl5zcf57b/weather_soil.RDS?raw=1", destfile = "weather_soil.RDS")
data = readRDS("weather_soil.RDS")

X = data$train # Features of the last 180 days
dim(X)
library(keras3)
Y = data$target
dim(Y)

plot(as.vector(Y[1:16,]), type = "l", xlab = "week", ylab = "Drought")

library(keras3)
holdout = 700:999
X_train = X[-holdout,,]
X_test = X[holdout,,] 
Y_train = Y[-holdout,]
Y_test = Y[holdout,]

model = keras_model_sequential(shape(dim(X)[2:3]))
model %>% 
  layer_lstm(units = 60L, activation = "relu") %>%  
# long short time memory cells
  layer_dense(units = 6L, activation = "linear")

summary(model)

model %>% compile(loss = loss_mean_absolute_error, optimizer = optimizer_adamax(0.0001))

model %>% fit(x = X_train, y = Y_train, epochs = 30L)


preds = model %>% predict(X_test)

matplot(cbind(as.vector(preds[1:48,]),  
              as.vector(Y_test[1:48,])), 
        col = c("darkblue", "darkred"),
        type = "o", 
        pch = c(15, 16),
        xlab = "week", ylab = "Drought")
legend("topright", bty = "n", 
       col = c("darkblue", "darkred"),
       pch = c(15, 16), 
       legend = c("Prediction", "True Values"))
