#' ---
#' title: "Kaggle - House Prices"
#' subtitle: "Step 3: Data Modeling"
#' author: "Gimme the Prize"
#' output: 
#'    html_document:
#'      theme: flatly
#'      toc: true
#'      highlight: haddock
#'      fig_width: 9
#'      fig_caption: false
#' ---


#+ echo = FALSE
#+ setup, message = FALSE, warning = FALSE
library(tidyverse)
library(caret)  # for machine learning workflow
library(scales)  # for plot formatting with currency

#' # Load Data
d.train <- readRDS(file = "./ames_02_train.RDS")
d.test <- readRDS(file = "./ames_02_test.RDS")


#' # Modeling
#' 
#' I will try regularization with ridge, lasso, and elastic net, and decision trees with boosting, random forest, and gradient descent.
#' To compare the various models, I'll split `train` into `train80` and `valid`.
set.seed(12345)
partition = createDataPartition(d.train$logSalePrice, p = 0.80, list = FALSE)
d.train80 <- d.train[partition, ]
d.valid <- d.train[-partition, ]
rm(partition)

#' I'll use the `caret` package for workflow, and use a common training control object to ensure an apple-apples comparison of the models.
model.trControl <- trainControl(method = "cv",
                                number = 10,
                                #                                index = createFolds(d.train80$logSalePrice, k = 10),  # specify the fold index
                                verboseIter = FALSE,  # don't print the results to the log
                                savePredictions = "final")  # saves predictions for the optimal tuning parameters

#' ## Regularization Models 
#' ### Ridge Regression

#' Perform cross-validation to optimize $\lambda$, specifying `alpha = 0` in a tuning grid for ridge regression.
model.ridge <- train(
  logSalePrice ~ . -Id -SalePrice,
  data = d.train80,
  method = "glmnet",
  metric = "RMSE", 
  preProcess = c("zv",  # remove and variables with zero variance (a single value)
                 "medianImpute",   # impute NAs with median (unnecesary - already imputed)
                 "center", "scale"  # standardize data for linear models "it just works better."
                 #                 "pca"  # reduce dimensions of nearly-zero variance columns
  ),  
  tuneGrid = expand.grid(
    .alpha = 0,  # optimize a ridge regression
    .lambda = seq(0, 1, length = 100)),  # create range by experiment to find metric's local min
  trControl = model.trControl
)
model.ridge
plot(model.ridge, main = "Ridge Regression Parameter Tuning", xlab = "lambda")
(perf.ridge <- postResample(pred = predict(model.ridge, newdata = d.valid), 
                            obs = d.valid$logSalePrice))

#' ### Lasso Regression
model.lasso <- train(
  logSalePrice ~ . -Id -SalePrice,
  data = d.train80,
  method = "glmnet",
  metric = "RMSE", 
  preProcess = c("zv",  # remove and variables with zero variance (a single value)
                 "medianImpute",   # impute NAs with median (unnecesary - already imputed)
                 "center", "scale"  # standardize data for linear models "it just works better."
                 #                 "pca"  # reduce dimensions of nearly-zero variance columns
  ),  
  tuneGrid = expand.grid(
    .alpha = 1,  # optimize a ridge regression
    .lambda = seq(0, 1, length = 100)),  # create range by experiment to find metric's local min
  trControl = model.trControl
)
model.lasso
plot(model.lasso, main = "Lasso Regression Parameter Tuning", xlab = "lambda")
(perf.lasso <- postResample(pred = predict(model.lasso, newdata = d.valid), 
                            obs = d.valid$logSalePrice))


#' ### Elastic Net Regression
model.elnet <- train(
  logSalePrice ~ . -Id -SalePrice,
  data = d.train80,
  method = "glmnet",
  metric = "RMSE", 
  preProcess = c("zv",  # remove and variables with zero variance (a single value)
                 "medianImpute",   # impute NAs with median (unnecesary - already imputed)
                 "center", "scale"  # standardize data for linear models "it just works better."
                 #                 "pca"  # reduce dimensions of nearly-zero variance columns
  ),  
  tuneGrid = expand.grid(
    .alpha = seq(0, 1, length = 10),  # optimize a ridge regression
    .lambda = seq(0, .5, length = 20)),  # create range by experiment to find metric's local min
  trControl = model.trControl
)
model.elnet
plot(model.elnet, main = "Lasso Regression Parameter Tuning", xlab = "lambda")
(perf.elnet <- postResample(pred = predict(model.elnet, newdata = d.valid), 
                            obs = d.valid$logSalePrice))

#' ## Decision Trees 

#' ### Bagged
#' (See discussion on pssing in matrix instead of formula here: https://stats.stackexchange.com/questions/115470/mtry-tuning-given-by-caret-higher-than-the-number-of-predictors)
model.bag = train(
  #  logSalePrice ~ . -Id -SalePrice,
  #  data = d.train80,
  d.train80[, -c(1, 2, 85)], d.train80[, 85],  # pass in matrix instead of formula
  method = "ranger",
  metric = "RMSE", 
  preProcess = c("zv",  # remove and variables with zero variance (a single value)
                 "medianImpute",   # impute NAs with median (unnecesary - already imputed)
                 "center", "scale"  # standardize data for linear models "it just works better."
                 #                 "pca"  # reduce dimensions of nearly-zero variance columns
  ),  
  tuneLength = 5,  # choose up to 5 combinations of tuning parameters
  trControl = model.trControl
)
model.bag
plot(model.bag, main = "Bagged Tree Parameter Tuning")
(perf.bag <- postResample(pred = predict(model.bag, newdata = d.valid), 
                          obs = d.valid$logSalePrice))


#' ### Random Forest
#' ### Boosting


#' ### Model Evaluation
#' 
rbind(perf.ridge, perf.lasso, perf.elnet, perf.bag)
models <- list(ridge = model.ridge, 
               lasso = model.lasso, 
               elnet = model.elnet,
               bag = model.bag)
resamples(models) %>% summary(metric = "RMSE")
bwplot(resamples(models), metric = "RMSE", main = "Model Comparison on Resamples")


#' ### Fit Final Model
d.final <- d.train %>% select(-"SalePrice") %>% data.frame()
model.final <- train(
  logSalePrice ~ . -Id,
  data = d.final,
  method = "glmnet",
  metric = "RMSE", 
  preProcess = c("zv",  # remove and variables with zero variance (a single value)
                 "medianImpute",   # impute NAs with median (unnecesary - already imputed)
                 "center", "scale"  # standardize data for linear models "it just works better."
                 #                 "pca"  # reduce dimensions of nearly-zero variance columns
  ),  
  tuneGrid = expand.grid(
    .alpha = model.elnet$bestTune$alpha,
    .lambda = model.elnet$bestTune$lambda),
  trControl = model.trControl
)
model.final
(perf.final <- postResample(pred = predict(model.final, newdata = d.valid), 
                            obs = d.valid$logSalePrice))

#' ### Create Submission
preds <- predict(model.final, newdata = d.test)
sub <- data.frame(Id = d.test$Id, SalePrice = exp(preds))
write.csv(sub, file = "./ames_03.csv", row.names = F)
