Homework 2
================
Stojsin, Rastko
February 3, 2020

```{r}
library('ElemStatLearn')  ## for 'prostate'
library('splines')        ## for 'bs'
library('dplyr')          ## for 'select', 'filter', and others
library('magrittr')       ## for '%<>%' operator
library('glmnet')         ## for 'glmnet'
```

```{r}
## loading prostate data
data('prostate')

## split prostate into testing and training subsets
prostate_train <- prostate %>%
  filter(train == TRUE) %>% 
  select(-train)
prostate_test <- prostate %>%
  filter(train == FALSE) %>% 
  select(-train)

## correlation of all variables in prostate data set
prostate_train %>% select(-lpsa) %>% cor(method = "pearson")
```

```{r}
## predict lpsa consider all other predictors
## linear model
fit <- lm(lpsa ~ ., data=prostate_train)

## functions to compute testing/training error w/lm
L2_loss <- function(y, yhat)
  (y-yhat)^2
error <- function(dat, fit, loss=L2_loss)
  mean(loss(dat$lpsa, predict(fit, newdata=dat)))
## train_error 
print(paste0("training error (lin model w/least squares): ",error(prostate_train, fit)))
## testing error
print(paste0("test error (lin model w/least squares): ",error(prostate_test, fit)))

```

```{r}
## use glmnet to fit ridge
## glmnet fits using penalized L2 loss
## first create an input matrix and output vector
form  <- lpsa ~ 0 + lcavol + lweight + age + lbph + lcp + pgg45 + svi + gleason
x_inp <- model.matrix(form, data=prostate_train)
y_out <- prostate_train$lpsa
fit <- glmnet(x=x_inp, y=y_out, lambda=seq(0.5, 0, -0.05), alpha = 0)
print(fit$beta)


## functions to compute testing/training error with glmnet
error <- function(dat, fit, lam, form, loss=L2_loss) {
  x_inp <- model.matrix(form, data=dat)
  y_out <- dat$lpsa
  y_hat <- predict(fit, newx=x_inp, s=lam)  ## see predict.elnet
  mean(loss(y_out, y_hat))
}
```