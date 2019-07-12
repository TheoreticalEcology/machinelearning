library(glmnet)


# Gaussian
x=matrix(rnorm(100*100),100,100)
effects = c(runif(30), rep(0,70))
effects = c(rep(1,15), rep(0,85))
y=rnorm(100) + x %*% effects

fit0 = lm(y ~ x)
summary(fit0)

fit1=glmnet(x,y)
plot(fit1)

cv = cv.glmnet(x,y)
plot(cv)

cv2 = cv.glmnet(x,y, alpha = 0.5)
plot(cv2)
