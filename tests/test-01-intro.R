
testthat::test_that('##  eval=knitr::is_html_output(excludes =  epub ), results =  asis , echo = F', {testthat::expect_error( {
## ---- eval=knitr::is_html_output(excludes = "epub"), results = 'asis', echo = F---------------------------------------------------
## cat(
## '<iframe width="560" height="315"
##   src="https://www.youtube.com/embed/1AVrWvRvfxs"
##   frameborder="0" allow="accelerometer; autoplay; encrypted-media;
##   gyroscope; picture-in-picture" allowfullscreen>
##   </iframe>'
## )


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## iris plot, fig.width=10, fig.height=4', {testthat::expect_error( {
## ----iris plot, fig.width=10, fig.height=4----------------------------------------------------------------------------------------
colors = hcl.colors(3)
traits = as.matrix(iris[,1:4]) 
species = iris$Species
image(y = 1:4, x = 1:length(species) , z = traits, 
      ylab = "Floral trait", xlab = "Individual")
segments(50.5, 0, 50.5, 5, col = "black", lwd = 2)
segments(100.5, 0, 100.5, 5, col = "black", lwd = 2)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
pairs(traits, pch = as.integer(species), col = colors[as.integer(species)])


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

#Reminder: traits = as.matrix(iris[,1:4])

d = dist(traits)
hc = hclust(d, method = "complete")

plot(hc)
rect.hclust(hc, k = 3)  #Draw rectangles around the branches.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(ape)
plot(as.phylo(hc), 
     tip.color = colors[as.integer(species)], 
     direction = "downwards")

hcRes3 = cutree(hc, k = 3)   #Cut a dendrogram tree into groups.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
tmp = hcRes3
tmp[hcRes3 == 2] = 3
tmp[hcRes3 == 3] = 2
hcRes3 = tmp
table(hcRes3, species)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
hc = hclust(d, method = "ward.D2")

plot(as.phylo(hc), 
     tip.color = colors[as.integer(species)], 
     direction = "downwards")

hcRes3 = cutree(hc, k = 3)   #Cut a dendrogram tree into groups.
table(hcRes3, species)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  results= hide , message=FALSE, warning=FALSE', {testthat::expect_error( {
## ---- results='hide', message=FALSE, warning=FALSE--------------------------------------------------------------------------------
library(dendextend)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
methods = c("ward.D", "single", "complete", "average",
             "mcquitty", "median", "centroid", "ward.D2")
out = dendlist()   #Create a dendlist object from several dendrograms.
for(i in seq_along(methods)) {
  res = hclust(d, method = methods[i])   
  out = dendlist(out, as.dendrogram(res))
}
names(out) = methods
print(out)

get_ordered_3_clusters = function(dend) {
  #order.dendrogram function returns the order (index)
  #or the "label" attribute for the leaves.
  cutree(dend, k = 3)[order.dendrogram(dend)]
}
dend_3_clusters = lapply(out, get_ordered_3_clusters)

compare_clusters_to_iris = function(clus){
  FM_index(clus, rep(1:3, each = 50), assume_sorted_vectors = TRUE)
}

clusters_performance = sapply(dend_3_clusters, compare_clusters_to_iris)
dotchart(sort(clusters_performance), xlim = c(0.3, 1),
         xlab = "Fowlkes-Mallows index",
         main = "Performance of linkage methods
         in detecting the 3 species \n in our example",
         pch = 19)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

#Reminder: traits = as.matrix(iris[,1:4]).

kc = kmeans(traits, 3)
print(kc)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
plot(iris[c("Sepal.Length", "Sepal.Width")],
     col =  colors[as.integer(species)], pch = kc$cluster)
points(kc$centers[, c("Sepal.Length", "Sepal.Width")],
       col = colors, pch = 1:3, cex = 3)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
table(iris$Species, kc$cluster)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=F', {testthat::expect_error( {
## ---- eval=F----------------------------------------------------------------------------------------------------------------------
## library(animation)
## saveGIF(kmeans.ani(x = traits[,1:2], col = colors),
##         interval = 1, ani.width = 800, ani.height = 800)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

getSumSq = function(k){kmeans(traits, k, nstart=25)$tot.withinss}
#Perform algorithm for different cluster sizes and retrieve variance.
iris.kmeans1to10 = sapply(1:10, getSumSq)
plot(1:10, iris.kmeans1to10, type="b", pch = 19, frame = FALSE, 
     xlab="Number of clusters K",
     ylab="Total within-clusters sum of squares",
     col = c("black", "red", rep("black", 8)))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

#Reminder: traits = as.matrix(iris[,1:4]).

library(dbscan)
kNNdistplot(traits, k =  4)   #Calculate and plot k-nearest-neighbor distances.
abline(h = 0.4, lty = 2)

dc = dbscan(traits, eps = 0.4, minPts = 6)
print(dc)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(factoextra)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  results= hide , message=FALSE, warning=FALSE', {testthat::expect_error( {
## ---- results='hide', message=FALSE, warning=FALSE--------------------------------------------------------------------------------
fviz_cluster(dc, traits, geom = "point", ggtheme = theme_light())


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(mclust)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  results= hide , message=FALSE, warning=FALSE', {testthat::expect_error( {
## ---- results='hide', message=FALSE, warning=FALSE--------------------------------------------------------------------------------
mb = Mclust(traits)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
mb$G # Two clusters.
mb$modelName # > Ellipsoidal, equal shape.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
mb3 = Mclust(traits, 3)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
plot(mb3, "density")


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
plot(mb3, what=c("classification"), add = T)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
table(iris$Species, mb3$classification)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
pcTraits = prcomp(traits, center = TRUE, scale. = TRUE)
biplot(pcTraits, xlim = c(-0.25,0.25), ylim = c(-0.25,0.25))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=knitr::is_html_output(excludes =  epub ), results =  asis , echo = F', {testthat::expect_error( {
## ---- eval=knitr::is_html_output(excludes = "epub"), results = 'asis', echo = F---------------------------------------------------
## cat(
##   '<iframe width="560" height="315"
##   src="https://www.youtube.com/embed/i04Pfrb71vk"
##   frameborder="0" allow="accelerometer; autoplay; encrypted-media;
##   gyroscope; picture-in-picture" allowfullscreen>
##   </iframe>'
## )


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  results= hide , message=FALSE, warning=FALSE', {testthat::expect_error( {
## ---- results='hide', message=FALSE, warning=FALSE--------------------------------------------------------------------------------
library(randomForest)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

m1 = randomForest(Sepal.Length ~ ., data = iris)   # ~.: Against all others.
# str(m1)
# m1$type
# predict(m1)
print(m1)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
par(mfrow = c(1, 2))
plot(predict(m1), iris$Sepal.Length,
     xlab = "Predicted", ylab = "Observed")
abline(0, 1)
varImpPlot(m1)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  message=FALSE, warning=FALSE', {testthat::expect_error( {
## ---- message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------
# devtools::install_github('araastat/reprtree')
reprtree:::plot.getTree(m1, iris)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

m1 = randomForest(Species ~ ., data = iris)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  message=FALSE, warning=FALSE', {testthat::expect_error( {
## ---- message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------
par(mfrow = c(1,2))
reprtree:::plot.getTree(m1, iris)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
oldpar = par(mfrow = c(1, 2))
plot(iris$Petal.Width, iris$Petal.Length, col = iris$Species, 
     main = "Observed")
plot(iris$Petal.Width, iris$Petal.Length, col = predict(m1),
     main = "Predicted")


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
par(oldpar)   #Reset par.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
table(predict(m1), iris$Species)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=knitr::is_html_output(excludes =  epub ), results =  asis , echo = F', {testthat::expect_error( {
## ---- eval=knitr::is_html_output(excludes = "epub"), results = 'asis', echo = F---------------------------------------------------
## cat(
##   '<iframe width="560" height="315"
##   src="https://www.youtube.com/embed/MotG3XI2qSs"
##   frameborder="0" allow="accelerometer; autoplay; encrypted-media;
##   gyroscope; picture-in-picture" allowfullscreen>
##   </iframe>'
## )


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(tensorflow)
# Don't worry about weird messages. TensorFlow supports additional optimizations.
exists("tf")

immutable = tf$constant(5.0)
mutable = tf$constant(5.0)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
a = tf$constant(5)
b = tf$constant(10)
print(a)
print(b)
c = tf$add(a, b)
print(c)
tf$print(c)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
`+.tensorflow.tensor` = function(a, b) return(tf$add(a,b))
tf$print(a+b)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
d = c + 5  # 5 is automatically converted to a tensor.
print(d)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
class(d)
class(d$numpy())


## int a = 5;

## float a = 5.0;

## char a = "a";


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## r_matrix = matrix(runif(10*10), 10, 10)
## m = tf$constant(r_matrix, dtype = "float32")
## b = tf$constant(2.0, dtype = "float64")
## c = m / b # Doesn't work! we try to divide float32/float64.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=TRUE', {testthat::expect_error( {
## ---- eval=TRUE-------------------------------------------------------------------------------------------------------------------
r_matrix = matrix(runif(10*10), 10,10)
m = tf$constant(r_matrix, dtype = "float64")
b = tf$constant(2.0, dtype = "float64")
c = m / b # Now it works.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
r_matrix = matrix(runif(10*10), 10, 10)
m = tf$constant(r_matrix, dtype = tf$float64)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## is.integer(5)
## is.integer(5L)
## matrix(t(r_matrix), 5, 20, byrow = TRUE)
## tf$reshape(r_matrix, shape = c(5, 20))$numpy()
## tf$reshape(r_matrix, shape = c(5L, 20L))$numpy()


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(torch)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
a = torch_tensor(1.)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
mutable = torch_tensor(5, requires_grad=TRUE) # tf$Variable(...)
immutable = torch_tensor(5, requires_grad=FALSE) # tf$constant(...)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
a = torch_tensor(5.)
b = torch_tensor(10.)
print(a)
print(b)
c = a$add(b)
print(c)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
a = torch_tensor(5.)
b = torch_tensor(10.)
print(a+b)
print(a/b)
print(a*b)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
d = a + 5  # 5 is automatically converted to a tensor.
print(d)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
class(d)
class(as.numeric(d))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  eval=FALSE', {testthat::expect_error( {
## ---- eval=FALSE------------------------------------------------------------------------------------------------------------------
## r_matrix = matrix(runif(10*10), 10,10)
## m = torch_tensor(r_matrix, dtype = torch_float32())
## b = torch_tensor(2.0, dtype = torch_float64())
## c = m / b


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(keras)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
library(keras)
library(tensorflow)

data(iris)
head(iris)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
X = scale(iris[,1:4])
Y = iris[,5]


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
Y = to_categorical(as.integer(Y)-1L, 3)
head(Y) # 3 colums, one for each level of the response.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
model = keras_model_sequential()


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
model %>%
  layer_dense(units = 20L, activation = "relu", input_shape = list(4L)) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 20L) %>%
  layer_dense(units = 3L, activation = "softmax") 


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
model_torch = 
  nn_sequential(
    nn_linear(4L, 20L),
    nn_linear(20L, 20L),
    nn_linear(20L, 20L),
    nn_linear(20L, 3L),
    nn_softmax(2)
  )


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
model %>%
  compile(loss = loss_categorical_crossentropy,
          keras::optimizer_adamax(lr = 0.001))
summary(model)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
optimizer_torch = optim_adam(params = model_torch$parameters, lr = 0.001)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
set.seed(123)

model_history =
  model %>%
    fit(x = X, y = apply(Y, 2, as.integer), epochs = 30L,
        batch_size = 20L, shuffle = TRUE)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
set.seed(123)

# Calculate number of training steps.
epochs = 30
batch_size = 20
steps = round(nrow(X)/batch_size*epochs)

X_torch = torch_tensor(X)
Y_torch = torch_tensor(apply(Y, 1, which.max)) 

# Set model into training status.
model_torch$train()

log_losses = NULL

# Training loop.
for(i in 1:steps){
  # Get batch.
  indices = sample.int(nrow(X), batch_size)
  
  # Reset backpropagation.
  optimizer_torch$zero_grad()
  
  # Predict and calculate loss.
  pred = model_torch(X_torch[indices, ])
  loss = nnf_cross_entropy(pred, Y_torch[indices])
  
  # Backpropagation and weight update.
  loss$backward()
  optimizer_torch$step()
  
  log_losses[i] = as.numeric(loss)
}


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
plot(model_history)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
plot(log_losses, xlab = "steps", ylab = "loss", las = 1)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
predictions = predict(model, X) # Probabilities for each class.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
head(predictions) # Quasi-probabilities for each species.


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
preds = apply(predictions, 1, which.max) 
print(preds)


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('## ', {testthat::expect_error( {
## ---------------------------------------------------------------------------------------------------------------------------------
model_torch$eval()
preds_torch = model_torch(torch_tensor(X))
preds_torch = apply(preds_torch, 1, which.max) 
print(preds_torch)
mean(preds_torch == as.integer(iris$Species))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
mean(preds == as.integer(iris$Species))


list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
testthat::test_that('##  cache=TRUE', {testthat::expect_error( {
## ---- cache=TRUE------------------------------------------------------------------------------------------------------------------
oldpar = par(mfrow = c(1, 2))
plot(iris$Sepal.Length, iris$Petal.Length, col = iris$Species,
     main = "Observed")
plot(iris$Sepal.Length, iris$Petal.Length, col = preds,
     main = "Predicted")
par(oldpar)   #Reset par.

list2env(as.list(environment()), envir = .GlobalEnv)
}, NA)})
