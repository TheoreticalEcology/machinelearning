# Typical ML workflow
## **** Objective of this lesson: learn the typical ML workflow, from data cleaning and exploration to model fitting  ****

library(keras)
library(tensorflow)
library(tidyverse)


# 1. Data input and cleaning - EDA (exploratory data analysis)

## Example - categorical variable
## The Titanic dataset is already split into a training (with information about the response) and a test dataset (w/o information about the response)
## But for data cleaning we have to combine the two datasetrs
load("titanic.RData")
test$survived = NA
train$subset = "train"
test$subset = "test"
data = rbind(train,test)


str(data)
summary(data)
head(data)

data$name

first_split = sapply(data$name, function(x) stringr::str_split(x, pattern = ",")[[1]][2])

titles = sapply(first_split, function(x) strsplit(x, ".",fixed = TRUE)[[1]][1])

table(titles)
titles = stringr::str_trim((titles))
titles %>%
  fct_count()

titles2 =
  forcats::fct_collapse(titles,
                        officer = c("Capt", "Col", "Major", "Dr", "Rev"),
                        royal = c("Jonkheer", "Don", "Sir", "the Countess", "Dona", "Lady"),
                        miss = c("Miss", "Mlle"),
                        mrs = c("Mrs", "Mme", "Ms")
                        )
titles2 %>%
  fct_count()

data =
  data %>%
    mutate(title = titles2)


## Example - numeric variable
summary(data)
sum(is.na(data$age))/nrow(data) # 20% NAs...

### We have to fix the NAs
data =
  data %>%
    group_by(sex, pclass, title) %>%
    mutate(age2 = ifelse(is.na(age), median(age, na.rm = TRUE), age)) %>%
    ungroup()




# 2. Pre-processing and feature selection

data_sub =
  data %>%
    select(survived, sex, age2, fare, title, pclass) %>%
    mutate(age2 = scales::rescale(age2, c(0,1)), fare = scales::rescale(fare, c(0,1))) %>%
    mutate(sex = as.integer(sex) - 1L, title = as.integer(title) - 1L, pclass = as.integer(pclass - 1L))

one_title = k_one_hot(data_sub$title, length(unique(data$title)))$numpy()
colnames(one_title) = levels(data$title)

one_sex = k_one_hot(data_sub$sex, length(unique(data$sex)))$numpy()
colnames(one_sex) = levels(data$sex)

one_pclass = k_one_hot(data_sub$pclass,  length(unique(data$pclass)))$numpy()
colnames(one_pclass) = paste0(1:length(unique(data$pclass)), "pclass")




data_sub = cbind(data.frame(survived= data_sub$survived, subset = data$subset), one_title, one_sex, age = data_sub$age2, fare = data_sub$fare, one_pclass)
head(data_sub)

### Split intro train and test:

train = data_sub %>% 
  filter(subset == "train") %>% 
  filter(!is.na(fare)) %>% 
  select(-subset)

indices = sample.int(nrow(train), 0.7*nrow(train))
sub_train = train[indices,]
sub_test = train[-indices,]


# 3. Model fitting

## 3.1 Fit on train


model = keras_model_sequential()
model %>%
  layer_dense(units = 20L, input_shape = ncol(sub_train) - 1L, activation = "relu") %>%
  layer_dense(units = 20L, activation = "relu") %>%
  layer_dense(units = 20L, activation = "relu") %>%
  layer_dense(units = 2L, activation = "softmax")
summary(model)

model_history =
model %>%
  compile(loss = loss_categorical_crossentropy, optimizer = optimizer_adamax(0.01))

model_history =
  model %>%
    fit(x = as.matrix(sub_train[,-1]), y = to_categorical(sub_train[,1],num_classes = 2L), epochs = 100L, batch_size = 32L, validation_split = 0.2, shuffle = TRUE)



# 4. Model evaluation
preds =
  model %>%
    predict(x = as.matrix(sub_test[,-1]))
predicted = ifelse(preds[,2] < 0.5, 0, 1)
observed = sub_test[,1]

(accuracy = mean(predicted == observed))




# 5. Predict and submit:
submit= 
  data_sub %>% 
    filter(subset == "test") %>% 
    select(-subset, -survived)

pred = model %>% 
  predict(as.matrix(submit))

pred = ifelse(pred[,2] < 0.5, 0, 1)
write.csv(data.frame(y=pred), file = "test_submit.csv")




# Exercise
# - Play around with the feature engineering, see https://www.kaggle.com/c/titanic/notebooks?sortBy=hotness&group=everyone&pageSize=20&competitionId=3136&language=R  for ideas
# - Play around with model parameters, optimizer(lr = ...), epochs = ..., number of hidden nodes in layers: units = ...
# - Try to maximize the model's accuarcy for the test data
# - Compare with randomForest


