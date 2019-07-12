# - preprocess data: cleaning, scaling, impute NA
# - fit svm, rf, brt and kknn on train
# - predict test and submit predictions


## Project 1 - Indian liver patient, 
# target var = Dataset, 
# https://www.kaggle.com/uciml/indian-liver-patient-records
data = readRDS(file = "Day2/Indian.RDS")
train = data$train
test = data$test

test$Dataset = NA
train$subset = "train"
test$subset = "test"
data = rbind(train, test)

data = data[,-1]
data[,1:10] = missRanger::missRanger(data[,1:10])
data[, which(sapply(data[,1:10], is.numeric), arr.ind = TRUE)] = scale(data[, which(sapply(data[,1:10], is.numeric), arr.ind = TRUE)])



train = data[data$subset== "train",-12]
test = data[data$subset== "test",-c(11,12)]


write.csv(data.frame(pred = ifelse(pred$data[,2] < 0.4, 0, 1)), file = "sub_max.csv")
# http://rhsbio6:8500 


## Project 2 - Exoplanet, 
# target var = LABEL



## Final project, case 1 - Kickstarter, target var = state (0,1 for failed/successful encoded)
# https://www.kaggle.com/kemical/kickstarter-projects/kernels