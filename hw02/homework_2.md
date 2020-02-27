Homework 2
================
Stojsin, Rastko
February 3, 2020

``` r
library('ElemStatLearn')  ## for 'prostate'
library('splines')        ## for 'bs'
library('dplyr')          ## for 'select', 'filter', and others
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
library('magrittr')       ## for '%<>%' operator
library('glmnet')         ## for 'glmnet'
```

    ## Loading required package: Matrix

    ## Loaded glmnet 3.0-1

``` r
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

    ##             lcavol    lweight       age        lbph        svi         lcp
    ## lcavol  1.00000000 0.30023199 0.2863243  0.06316772  0.5929491  0.69204308
    ## lweight 0.30023199 1.00000000 0.3167235  0.43704154  0.1810545  0.15682859
    ## age     0.28632427 0.31672347 1.0000000  0.28734645  0.1289023  0.17295140
    ## lbph    0.06316772 0.43704154 0.2873464  1.00000000 -0.1391468 -0.08853456
    ## svi     0.59294913 0.18105448 0.1289023 -0.13914680  1.0000000  0.67124021
    ## lcp     0.69204308 0.15682859 0.1729514 -0.08853456  0.6712402  1.00000000
    ## gleason 0.42641407 0.02355821 0.3659151  0.03299215  0.3068754  0.47643684
    ## pgg45   0.48316136 0.07416632 0.2758057 -0.03040382  0.4813577  0.66253335
    ##            gleason       pgg45
    ## lcavol  0.42641407  0.48316136
    ## lweight 0.02355821  0.07416632
    ## age     0.36591512  0.27580573
    ## lbph    0.03299215 -0.03040382
    ## svi     0.30687537  0.48135774
    ## lcp     0.47643684  0.66253335
    ## gleason 1.00000000  0.75705650
    ## pgg45   0.75705650  1.00000000

``` r
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
```

    ## [1] "training error (lin model w/least squares): 0.439199768058334"

``` r
## testing error
print(paste0("test error (lin model w/least squares): ",error(prostate_test, fit)))
```

    ## [1] "test error (lin model w/least squares): 0.521274005650894"

``` r
## use glmnet to fit ridge
## glmnet fits using penalized L2 loss
## first create an input matrix and output vector
form  <- lpsa ~ 0 + lcavol + lweight + age + lbph + lcp + pgg45 + svi + gleason
x_inp <- model.matrix(form, data=prostate_train)
y_out <- prostate_train$lpsa
fit <- glmnet(x=x_inp, y=y_out, lambda=seq(0.5, 0, -0.05), alpha = 0)
print(fit$beta)
```

    ## 8 x 11 sparse Matrix of class "dgCMatrix"

    ##    [[ suppressing 11 column names 's0', 's1', 's2' ... ]]

    ##                                                                          
    ## lcavol   0.330033285  0.341589837  0.3544771780  0.368972918  0.385350033
    ## lweight  0.515702133  0.525401042  0.5354348208  0.545774936  0.556333416
    ## age     -0.004889214 -0.005636617 -0.0064641968 -0.007389019 -0.008433635
    ## lbph     0.111247895  0.113960541  0.1167938799  0.119761396  0.122899867
    ## lcp      0.015113718  0.008036700 -0.0002374648 -0.010099378 -0.022038917
    ## pgg45    0.004390006  0.004514712  0.0046621277  0.004839771  0.005060079
    ## svi      0.541703554  0.553405591  0.5658135655  0.579176717  0.593901939
    ## gleason  0.063402681  0.061363090  0.0588887163  0.055895060  0.052234345
    ##                                                                         
    ## lcavol   0.404130918  0.426074441  0.452123247  0.483787288  0.523682897
    ## lweight  0.567063929  0.577889623  0.588568720  0.598709576  0.607672220
    ## age     -0.009618197 -0.010967693 -0.012522584 -0.014334831 -0.016464993
    ## lbph     0.126206606  0.129663881  0.133291299  0.137089205  0.140983846
    ## lcp     -0.036651184 -0.054804333 -0.077864474 -0.107951309 -0.148533504
    ## pgg45    0.005338760  0.005700404  0.006186666  0.006865954  0.007867088
    ## svi      0.610203738  0.628338820  0.649022507  0.673210509  0.701932250
    ## gleason  0.047604314  0.041500842  0.033144271  0.021129661  0.002465277
    ##                     
    ## lcavol   0.576361989
    ## lweight  0.614010456
    ## age     -0.019004296
    ## lbph     0.144877454
    ## lcp     -0.206098858
    ## pgg45    0.009452544
    ## svi      0.737221825
    ## gleason -0.029176315

``` r
## functions to compute testing/training error with glmnet
error <- function(dat, fit, lam, form, loss=L2_loss) {
  x_inp <- model.matrix(form, data=dat)
  y_out <- dat$lpsa
  y_hat <- predict(fit, newx=x_inp, s=lam)  ## see predict.elnet
  mean(loss(y_out, y_hat))
}
```

``` r
## plot path diagram
plot(x=range(fit$lambda),
     y=range(as.matrix(fit$beta)),
     type='n',
     xlab=expression(lambda),
     ylab='Coefficients')
for(i in 1:nrow(fit$beta)) {
  points(x=fit$lambda, y=fit$beta[i,], pch=19, col='#00000055')
  lines(x=fit$lambda, y=fit$beta[i,], col='#00000055')
}
abline(h=0, lty=3, lwd=2)
```

![](homework_2_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

``` r
## compute training and testing errors as function of lambda
err_train <- sapply(fit$lambda, function(lam) 
  error(prostate_train, fit, lam, form))
err_test <- sapply(fit$lambda, function(lam) 
  error(prostate_test, fit, lam, form))

## plot test/train error
plot(x=range(fit$lambda),
     y=range(c(err_train, err_test)),
     type='n',
     xlab=expression(lambda),
     ylab='train/test error')
points(fit$lambda, err_train, pch=19, type='b', col='darkblue')
points(fit$lambda, err_test, pch=19, type='b', col='darkred')
legend('topleft', c('train','test'), lty=1, pch=19,
       col=c('darkblue','darkred'), bty='n')
```

![](homework_2_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->
