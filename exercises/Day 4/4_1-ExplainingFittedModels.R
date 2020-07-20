set.seed(123)
library("iml")
library("randomForest")
data("Boston", package  = "MASS")
rf = randomForest(medv ~ ., data = Boston, ntree = 50)


# We create a Predictor object, that holds the model and the data. The iml package uses R6 classes: New objects can be created by calling Predictor$new().

X = Boston[which(names(Boston) != "medv")]
predictor = Predictor$new(rf, data = X, y = Boston$medv)

imp = FeatureImp$new(predictor, loss = "mae")
plot(imp)

# Accumuluated local effects and partial dependence plots both show the average model prediction over the feature. The difference is that ALE are computed as accumulated differences over the conditional distribution and partial dependence plots over the marginal distribution. ALE plots preferable to PDPs, because they are faster and unbiased when features are correlated.

ale = FeatureEffect$new(predictor, feature = "lstat")
ale$plot()

# Again, but this time with a partial dependence plot and ice curves
eff = FeatureEffect$new(predictor, feature = "rm", method = "pdp+ice", grid.size = 30)
plot(eff)

# plot all effects
effs = FeatureEffects$new(predictor)
plot(effs)

interact = Interaction$new(predictor, "lstat")
plot(interact)

# global surrogate 

tree = TreeSurrogate$new(predictor, maxdepth = 2)
plot(tree)

# local model
# LocalModel fits locally weighted linear regression models (logistic regression for classification) to explain single predictions of a prediction model.

lime.explain = LocalModel$new(predictor, x.interest = X[1,])
lime.explain$results
plot(lime.explain)

# Shapley computes feature contributions for single predictions with the Shapley value, an approach from cooperative game theory. The features values of an instance cooperate to achieve the prediction. The Shapley value fairly distributes the difference of the instance's prediction and the datasets average prediction among the features.

shapley = Shapley$new(predictor, x.interest = X[1,])
shapley$plot()
