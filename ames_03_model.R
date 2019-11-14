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


#' I will try linear regression with ols, and stepwise, regularization with 
#' ridge, lasso, and elastic net, and decision trees with boosting, random 
#' forest, and gradient descent.  

#' # Setup

#+ echo = FALSE
#+ setup, message = FALSE, warning = FALSE
library(tidyverse)
library(caret)  # for machine learning workflow
library(scales)  # for plot formatting with currency
library(skimr)
library(broom)

d.train <- readRDS(file = "./ames_02_train.RDS")
d.test <- readRDS(file = "./ames_02_test.RDS")


#' To compare the various models, I'll split 
#' `train` into `train80` and `valid`.
#'
set.seed(12345)
partition = createDataPartition(d.train$logSalePrice, p = 0.80, list = FALSE)
d.train80 <- d.train[partition, ]
d.valid <- d.train[-partition, ]
rm(partition)

#' I'll use the `caret` package for workflow, and use a common training 
#' control object to ensure an apple-apples comparison of the models. I will
#' use 10-fold cross-validation.
#' 
model.trControl <- trainControl(method = "cv",
                                # specify the 10 folds instead of n = 10
                                index = createFolds(d.train80$logSalePrice, 
                                                    k = 10, 
                                                    returnTrain = TRUE),  
                                verboseIter = TRUE,  # print results to log?
                                # save predictions for optimal tuning params
                                savePredictions = "final")  

#' # Linear Models
#' 
#' I'll start with a straight linear regression, throwing in all the 
#' predictors, then
#' I'll try the stepwise model selection methods (forward-selection, backward-
#' selection, and stepwise selection).
#' 
#' ## lm
#' 
#' Near-zero variance (NZV) predictors may become zero-variance predictors when 
#' the data are split into cross-validation sub-samples, and they may cause a 
#' few samples to have an undue influence on the model. I will remove them prior 
#' to modeling.  Here are the NZVs in the training data set.
nzv <- nearZeroVar(d.train80)
colnames(d.train80[, nzv])
d.train80.nzv <- d.train80[, -nzv]

#' Fit the model, removing near-zero value predictors, then Box-Cox 
#' transforming predictors to correct for skew.  I'll also center and
#' scale the data so I can evaluate the relative influences of the predictors.
#' 
model.lm <- train(
  logSalePrice ~ . -Id -SalePrice,
  data = d.train80.nzv,
  method = "lm",
  metric = "RMSE",
  preProcess = c("nzv", "BoxCox", "center", "scale"),
  trControl = model.trControl
)
print(model.lm)

#' The Box-Cox procedure only applies to variables with values >0.  Only 
#' 12 such variables were transformed.  (I tried this with a Yeo-Johnson
#' transformation, but it produced a poorer performing model, so I'm staying
#' with Box-Cox.)
names(model.lm$preProcess$bc)

#' The within-sample RMSE was 0.1720.  (The within-sample RMSE is the average
#' of 10 hold-out fold RMSEs, `mean(model.lm$resample$RMSE)`).  Here are the 
#' results of the individual folds.
#'  
#+ fig.height = 3
model.lm$resample %>%
  ggplot(aes(x = Resample, y = RMSE)) +
  geom_col(fill = "cadetblue") +
  theme_minimal() +
  labs(title = "lm model CV RMSE")

#' It looks like there might be some predictions that are very, very 
#' inaccurate.  Here is an observed vs predicted plot with prediction errors
#' >10% labeled.
#' 
model.lm$pred %>% 
  mutate(label = ifelse(abs(obs - pred) / obs > 0.10, rowIndex, "")) %>%
  ggplot(aes(x = pred, y = obs, label = label)) + 
  geom_point(alpha = .2, color = "cadetblue") + 
  geom_abline(slope = 1, intercept = 0, color = "cadetblue", linetype = 2) +
  geom_text(size = 3, position = position_jitter(width = .3, height = .3)) +
  expand_limits(x = c(7.5, 15), y = c(7.5, 15)) +
  labs(title = "LM: Observed vs Predicted")

#' What I'd like to do now is learn which features caused these outliers.
#' One way to do that is to multiply each feature by the corresponding model
#' coefficient estimate.  I'm not sure how to do that since I didn't use a
#' model matrix and `lm()` automatically expands factors into dummies.  Plus,
#' for the ordinal variables, I need to convert the value into a polynomial
#' contrast.  And I've centered and scaled my variables, so I need to back that
#' out.  Blech.  My original plan (before noticing these outliers) 
#' was to identify which predictors were both significant and meaningful.  I
#' standardized the predictors in the model, so I can do this by must comparing 
#' the coefficients.  The coefficient values indicate the effect of a 1 SD 
#' change in the predictor value on the log SalePrice.
#' Here are the predictors with p-values <.05 and estimates > 0.1.
summary(model.lm)$coefficients %>%
  data.frame() %>%
  rownames_to_column(var = "Variable") %>%
  mutate(p = `Pr...t..`) %>%
  mutate(est2 = exp(Estimate)) %>%
  filter(abs(Estimate) > .01 & p < .05 & Variable != "(Intercept)") %>%
  select(Variable, Estimate, p, est2) %>%
  arrange(desc(abs(Estimate))) %>%
  top_n(10)

#' This is interesting in its own right, but doesn't help me understand why my
#' model produced those outragous prediction errors in the CV.

#' Moving on, how does my model perform against the validation set?
#' 
(perf.lm <- postResample(pred = predict(model.lm, newdata = d.valid), 
                         obs = d.valid$logSalePrice))

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
rbind(perf.lm, perf.ridge, perf.lasso, perf.elnet, perf.bag)
models <- list(lm = model.lm,
               ridge = model.ridge, 
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
