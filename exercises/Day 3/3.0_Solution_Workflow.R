library(mlr)
library(keras)
library(tidyverse)

data = EcoData::titanic
data = data %>% 
  select(pclass, survived, sex, age, sibsp, parch, fare, body)

data[,-2] = missRanger::missRanger(data[,-2])

# define ml algorithm, see mlr website for supported learners
learner = makeLearner("classif.ranger", importance = "impurity", predict.type = "prob")
# whhich parameters can we tune:
getParamSet(learner)
# define hyper parameter space
param = makeParamSet(makeIntegerParam("mtry", 1L, 5L), 
                     makeIntegerParam("min.node.size", 2L, 50L))
# create classification task
task = makeClassifTask(data = data.frame(data), target = "survived", positive = "1")
# one hot encode all cateogircal columns
task = mlr::createDummyFeatures(task)
# scale all features
task = mlr::normalizeFeatures(task)
# balance classes
task = oversample(task, rate = 2)

# we tune in 40 random steps
control = makeTuneControlRandom(maxit = 40L)
# update learner object (inner validation)
learnerTune = makeTuneWrapper(learner, resampling = cv10, measures = auc, par.set = param, control = control)
# outer validation
resample = resample(learnerTune, task, cv5, measures = auc, models = TRUE, keep.pred = TRUE, extract = mlr::getTuneResult)


