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


#' My plan is to fit linear regression models ordinary least squares and 
#' partial least squares, penalized models ridge, lasso, 
#' and elastic net, tree-based models bagged, random forest, 
#' and boosting.
#' 
#' # Setup
#' 
#' Load the libraries and the data sets produced in the prior step.
 
#+ echo = FALSE
#+ setup, message = FALSE, warning = FALSE
library(tidyverse)
library(caret)  # for machine learning workflow
library(broom)
library(e1071)  # for Skewness
library(recipes)

#' Load the train and test data sets from the prior step.
#' 
d.train <- readRDS(file = "./ames_02_train.RDS")
d.test <- readRDS(file = "./ames_02_test.RDS")

#' # Linear Regression
#' 
#' All linear models are of the form
#' 
#' $$y_i = \beta X + \epsilon$$
#' 
#' and estimate the $\beta$ coefficients by minimizing the sum of squared 
#' errors, or minimizing a function of the squared errors.
#' 
#' Linear models are relatively easy to interpret, and the coefficient standard 
#' errors it generates enables inferential analysis.  A good framework for
#' thinking about the various linear models is how they handle the 
#' bias-variance trade-off.  Ordinary least squares (OLS) and partial least 
#' squares (PLS) parameter estimates *minimize bias*. Penalization models such as
#' ridge, lasso, and elastic net introduce some bias in order to *reduce variance*.
#'
#' I'll start with OLS regression.  After fitting an initial model,
#' I'll use the stepwise regression algorithm to reduce the predictor variable
#' set to a parsimonious model.  If the peformances are comparable, the 
#' parsimonious model would be attractive for real-world application.  Next I 
#' will try a PLS regression.  PLS is designed to manage covariance in the 
#' predictors.  Finally, I'll try the penalization algorithms, ridge, lasso, 
#' and elastic net.
#' 
#' ## OLS
#' 
#' OLS regression estimates the model coefficients by minimizing 
#' the sum of squared errors,
#'  
#' $$SSE = \sum{(y_i - \hat{y}_i)^2}.$$
#' 
#' The solution is $\hat{\beta} = (X'X)^{-1}X'y$. The OLS model is the best 
#' linear unbiased estimator if the residuals are independent random variables 
#' normally distributed with mean zero and constant variance.  In addition to 
#' validating these model assumptions, I need to be wary of multicollinearity.  
#' A unique inverse $(X'X)^{-1}$ only exists when there is no perfect 
#' multicollinearity and the matrix is of full rank, but even when predictors 
#' are only *highly* correlated, the regression coefficient estimations will 
#' still be unstable and can produce a poorly performing model.  
#' 
#' Predictive models are also vulnerable to near-zero variance (NZV) 
#' predictors.  NZV predictors contain little information, so they usually do
#' not add much value to a linear regression.  When cross-validating a model, 
#' NZV predictors can devolve into zero-variance predictors
#' in the individual folds and break the model.  
#'
#' I'll start by addressing NZV.  I'll collapsing factor levels so that there
#' are less NZV predictors after they are coded into dummy variables.  This is 
#' more or less trial and error with the aid of the box-plots and anova post-hoc 
#' testing from the exploratory phase.  The collapsing choices below improved
#' the RMSE in the holdout.

#' Bind the data sets back together for model-specific data engineering.
#' 
d <- bind_rows(d.train, d.test)

d_lin <- d %>%
  mutate(Alley = fct_lump(Alley, n = 1)) %>%
  mutate(BldgType = fct_recode(BldgType, T_1 = "TwnhsE", T_1 = "1Fam", D_T_2 = "2fmCon", D_T_2 = "Duplex", D_T_2 = "Twnhs")) %>%
  mutate(BsmtFinType1 = fct_recode(BsmtFinType1, Rec2BLQ = "Rec", Rec2BLQ = "BLQ", LwQALQ = "LwQ", LwQALQ = "ALQ")) %>%
  mutate(BsmtFinType1 = fct_relevel(BsmtFinType1, "NA", "Rec2BLQ", "LwQALQ", "Unf", "GLQ")) %>%
  mutate(BsmtFinType2 = fct_recode(BsmtFinType2, Rec2BLQ = "Rec", Rec2BLQ = "BLQ", LwQALQ = "LwQ", LwQALQ = "ALQ")) %>%
  mutate(BsmtFinType2 = fct_relevel(BsmtFinType2, "NA", "Rec2BLQ", "LwQALQ", "Unf", "GLQ")) %>%
  mutate(BsmtQual = fct_drop(BsmtQual)) %>%
  mutate(Electrical = fct_recode(Electrical, Other = "FuseP", Other = "FuseF", Other = "FuseA", Other = "Mix")) %>%
  mutate(ExterCond = fct_recode(ExterCond, Po_Fa = "Po", Po_Fa = "Fa", Ta_Ex = "TA", Ta_Ex = "Gd", Ta_Ex = "Ex")) %>%
  mutate(Exterior1st = fct_recode(Exterior1st, VinalSdP = "VinylSd", VinalSdP = "Stone", VinalSdP = "ImStucc",
                                  HdBoardP = "HdBoard", HdBoardP = "CBlock", HdBoardP = "AsphShn", HdBoardP = "BrkComm")) %>%
  mutate(Exterior2nd = fct_recode(Exterior2nd, WdSdngP = "Wd Sdng", WdSdngP = "WdShing", WdSdngP = "Stucco",
                                  HdBoardP = "HdBoard", HdBoardP = "ImStucc")) %>%
  mutate(ExterQual = fct_recode(ExterQual, Fa_TA = "Fa", Fa_TA = "TA")) %>%
  mutate(Fence = fct_recode(Fence, NA_GP = "NA", NA_GP = "GdPrv",
                            M_M_G = "MnWw", M_M_G = "MnPrv", M_M_G = "GdWo")) %>%
  mutate(FireplaceQu = fct_recode(FireplaceQu, NA_Fa = "NA", NA_Fa = "Po", NA_Fa = "Fa")) %>%
  mutate(Foundation = fct_recode(Foundation, CBlockP = "CBlock", CBlockP = "Stone")) %>%
  mutate(Functional = fct_lump(Functional, n = 1)) %>%
  mutate(GarageCond = fct_recode(GarageCond, NA_Fa = "NA", NA_Fa = "Po", NA_Fa = "Fa", 
                                 TA_Ex = "TA", TA_Ex = "Gd", TA_Ex = "Ex")) %>%
  mutate(GarageFinish = fct_recode(GarageFinish, NA_Unf = "NA", NA_Unf = "Unf")) %>%
  mutate(GarageQual = fct_recode(GarageQual, Po_Fa = "Po", Po_Fa = "Fa", Gd_Ex = "Gd", Gd_Ex = "Ex")) %>%
  mutate(GarageType = fct_recode(GarageType, CarPortP = "CarPort", CarPortP = "NA")) %>%
  mutate(Heating = fct_lump(Heating, n = 1)) %>%
  mutate(HouseStyle = fct_recode(HouseStyle, Story2P = "2Story", Story2P = "2.5Fin")) %>%
  mutate(LandContour = fct_lump(LandContour, n = 1)) %>%
  mutate(LandSlope = fct_lump(LandSlope, n = 1)) %>%
  mutate(LotShape = fct_recode(LotShape, IR = "IR1", IR = "IR2", IR = "IR3")) %>%
  mutate(MiscFeature = fct_lump(MiscFeature, n = 1)) %>%
  mutate(MSZoning = fct_recode(MSZoning, C_P = "C", C_P = "RH", C_P = "RM")) %>%
  mutate(Neighborhood = fct_recode(Neighborhood, A1 = "NoRidge", A1 = "NridgHt", A1 = "StoneBr",
                                   A2 = "Veenker", A2 = "Timber", A2 = "Somerst", A2 = "ClearCr",
                                   A3 = "Crawfor", A3 = "CollgCr", A3 = "Gilbert",
                                   A4 = "NWAmes", A4 = "Blmngtn", A4 = "SawyerW",
                                   A5 = "NAmes", A5 = "NPkVill", A5 = "Blueste", A5 = "Sawyer", A5 = "SWISU", A5 = "Mitchel",
                                   A6 = "OldTown", A6 = "BrkSide", A6 = "Edwards", A6 = "BrDale", A6 = "MeadowV", A6 = "IDOTRR")) %>%
  select(-SalePrice, -Condition1, -Condition2, -CondRoad, -CondRail, -CondPos)

#' Split the data back into train and test.  
#' 
d_lin_test <- d_lin[is.na(d_lin$logSalePrice), ]
d_lin_train <- d_lin[!is.na(d_lin$logSalePrice), ]

#' Further split the training data 80:20 into `d_lin_train80` (n = 1,161) and 
#' `d_lin_train20` (n = 289). I'll use `d_lin_train80` for tuning,  estimating 
#' coefficients, and estimating performance, and `d_lin_train20` to 
#' compare the resulting models. `createDataPartition()` ensures
#' the distribution of the response variable is similar in the two 
#' distributions.
#'
set.seed(12345)
partition <- createDataPartition(d_lin_train$logSalePrice, p = 0.80, list = FALSE)
d_lin_train80 <- d_lin_train[partition, ]
d_lin_train20 <- d_lin_train[-partition, ]

#' A common training control object will ensure each model is generated under
#' similar conditions.  I will use 10-fold cross-validation.  `createFolds()` 
#' creates folds with similar distributions of the target variable.
#' 
model.trControl <- trainControl(method = "cv",
                                # specify the 10 folds instead of n = 10
                                index = createFolds(d_lin_train80$logSalePrice, 
                                                    k = 10, 
                                                    returnTrain = TRUE),  
                                verboseIter = TRUE,  # print results to log?
                                # save predictions for optimal tuning params
                                savePredictions = "final")  


#' Create a recipe object to contol preprocessing.  Here is the basic recipe.
#' 
ols_recipe <- d_lin_train80 %>%
  recipe(logSalePrice ~ .) %>%
  update_role(Id, new_role = "ID")

#' Convert the nominal factor variables to dummy variables.  This balloons the 
#' predictor set up to 237.
#' 
ols_recipe_2 <- ols_recipe %>%
  step_dummy(all_nominal()) 
ols_baked_2 <- ols_recipe_2 %>% prep(d_lin_train80) %>% bake(d_lin_train80)
ols_baked_2 %>% names() %>% length() - 2

#' Remove the NZVs. 
#' 
nearZeroVar(ols_baked_2, saveMetrics = TRUE) %>% 
  rownames_to_column(var = "predictor") %>% 
  filter(nzv == TRUE)
#' 93 predictors are NZV.  `step_nzv()` removes them.  
ols_recipe_3 <- ols_recipe_2 %>%
  step_nzv(all_predictors()) 
ols_baked_3 <- ols_recipe_3 %>% prep(d_lin_train80) %>% bake(d_lin_train80)
ols_baked_3 %>% names() %>% length() - 2
#' That leaves 144 predictors.

#' Remove predictors that are exact linear combinations of other predictors.
#' 
lin_combos <- findLinearCombos(ols_baked_3)
map(lin_combos$linearCombos, function(x) colnames(ols_baked_3)[x])
colnames(ols_baked_3)[lin_combos$remove]
#' 10 predictors that are linear combinations.  `step_lincomb` removes them.
ols_recipe_4 <- ols_recipe_3 %>%
  step_lincomb(all_numeric(), -all_outcomes()) 
ols_baked_4 <- ols_recipe_4 %>% prep(d_lin_train80) %>% bake(d_lin_train80)
ols_baked_4 %>% names() %>% length() - 2
#' That leaves 134 predictors.

#' Remove predictors that have large (>=0.90) absolute correlations with the 
#' other predictors
#' 
cor_vars <- findCorrelation(cor(ols_baked_4), cutoff = .90)
colnames(ols_baked_4)[cor_vars]
#' 8 predictors are highly correlated with other predictors.  `step_corr` 
#' removes them.
ols_recipe_5 <- ols_recipe_4 %>%
  step_corr(all_numeric(), -all_outcomes(), threshold = .90) 
ols_baked_5 <- ols_recipe_5 %>% prep(d_lin_train80) %>% bake(d_lin_train80)
ols_baked_5 %>% names() %>% length() - 2
#' That leaves 126 predictors.

#' Box-Cox transformations are usually applied to the response variable, but
#' there is the option of applying it to the predictor variables too.  I'm not
#' sure why this makes sense, but it did help the model performance in this 
#' case, so I'll include it (reluctantly).  Here are the predictors (excluding
#' dummies) with significant (>=0.5) skew.
col_is_num <- map(d_lin_train80, ~ class(.x)[1] %in% c("integer", "numeric")) %>% unlist() 
col_skew <- map(d_lin_train80[, col_is_num], skewness) %>% unlist()
data.frame(var = names(col_skew), skew = col_skew) %>% 
  filter(abs(skew) > 0.5) %>%
  arrange(-abs(skew))
#' Box-Cox will transform strictly positive variables.
boxcox_vars <- preProcess(as.data.frame(ols_baked_5), method = "BoxCox")
boxcox_vars$bc %>% names()
#' 9 predictors (Id and logSalePrice excluded) have positive values that can
#' be transformed.
ols_recipe_6 <- ols_recipe_5 %>%
  step_BoxCox(all_numeric(), -all_outcomes())
ols_baked_6 <- ols_recipe_6 %>% prep(d_lin_train80) %>% bake(d_lin_train80)

#' Centering and scaling the data will not affect the model performance, but it
#' provides a useful way to compare the predictor affects on the response.
#' 
ols_recipe_7 <- ols_recipe_6 %>%
  step_center(all_predictors(), -all_nominal()) %>%
  step_scale(all_predictors(), -all_nominal()) 

#' Now train the model.
#' 
model.lm <- train(
  x = ols_recipe_7,
  data = d_lin_train80,
  method = "lm",
  metric = "RMSE",
  trControl = model.trControl
)

#' What did I get for a model?
#' 
summary(model.lm)

#' lm warned "X not defined because of singularities" prior to collapsing some
#' of the factor variables.  No errors now. How did the model perform in terms 
#' of the within-sample RMSE?
#' 
print(model.lm)

#' The within-sample RMSE was `r round(model.lm$results$RMSE, 4)`.  (The 
#' within-sample RMSE is the average
#' of 10 hold-out fold RMSEs, `mean(model.lm$resample$RMSE)`).  Here are 
#' the results of the individual folds.
#'  
#+ fig.height = 3
model.lm$resample %>%
  ggplot(aes(x = Resample, y = RMSE)) +
  geom_col(fill = "cadetblue") +
  theme_minimal() +
  labs(title = "lm model CV RMSE")

#' Here is an observed vs predicted plot with prediction errors
#' >10% labeled (none are).
#' 
model.lm$pred %>% 
  mutate(label = ifelse(abs(obs - pred) / obs > 0.10, rowIndex, "")) %>%
  ggplot(aes(x = pred, y = obs, label = label)) + 
  geom_point(alpha = .2, color = "cadetblue") + 
  geom_abline(slope = 1, intercept = 0, color = "cadetblue", linetype = 2) +
  geom_text(size = 3, position = position_jitter(width = .3, height = .3)) +
  expand_limits(x = c(7.5, 15), y = c(7.5, 15)) +
  labs(title = "LM: Observed vs Predicted")

#' Does the model conform to the linear model assumption that residuals are 
#' independent random variables normally distributed with mean zero and 
#' constant variance?  A standardized residuals vs fits plot vary around e=0
#' (linearity) with a constant width (especially no fan shape at the low or 
#' high ends) (equal variance).  95% of standardized residuals should fall 
#' within two standard deviations.
#' 
model.lm$pred %>% 
  mutate(res = pred-obs,
         stdres = res / sd(res),
         label = ifelse(abs(stdres) > 2, rowIndex, "")) %>%
  ggplot(aes(x = pred, y = stdres, label = label)) + 
  geom_point(alpha = .2, color = "cadetblue") + 
  geom_abline(slope = 0, intercept = 2, color = "cadetblue", linetype = 2) +
  geom_abline(slope = 0, intercept = -2, color = "cadetblue", linetype = 2) +
  geom_text(size = 2, position = position_jitter(width = .3, height = .3), color = "cadetblue") +
#  expand_limits(x = c(7.5, 15), y = c(7.5, 15)) +
  labs(title = "LM: Standardized Residuals vs Predicted")

#' A normal probability plot compares the theoretical percentiles of the normal 
#' distribution versus the observed sample percentiles. It should be 
#' approximately linear.  
stdres <- model.lm$pred %>% 
  mutate(res = pred-obs,
         stdres = res / sd(res)) %>%
  pull(stdres)
qqnorm(stdres, main = "LM: Normal Q-Q Plot")
qqline(stdres)

#' This one does not look so good.  The distribution is heavy tailed - There 
#' are many extreme positive and negative residuals.  The Anderson-Darling 
#' normality test p-value is the probability of calculating the test statistic 
#' if the distribution is normal.  This one is not normal.
library(nortest)
model.lm$pred %>% 
  mutate(res = pred-obs) %>%
  pull(res) %>% 
  ad.test()

#' Which predictors were both significant and meaningful?  I standardized the 
#' predictors in the model, so I can check this by just comparing 
#' the coefficients.  The coefficient values indicate the effect of a 1 SD 
#' change in the predictor value on the log SalePrice.
#' Here are the predictors with p-values <.05 and estimates > 0.1.
#' 
summary(model.lm)$coefficients %>%
  data.frame() %>%
  rownames_to_column(var = "Variable") %>%
  mutate(p = `Pr...t..`) %>%
  mutate(est2 = exp(Estimate)) %>%
  filter(abs(Estimate) > .01 & p < .05 & Variable != "(Intercept)") %>%
  select(Variable, Estimate, p, est2) %>%
  arrange(desc(abs(Estimate))) %>%
  top_n(10)

#' Overall quality (no surprise), followed by basement size (a bit of a 
#' surprise), then second and first floor size.  These all seem good.  Okay, 
#' moving on, how does the model perform against the validation set?
#' 
(perf.lm <- postResample(pred = predict(model.lm, newdata = d_lin_train20), 
                         obs = d_lin_train20$logSalePrice))

#' It's slightly better than the within-sample RMSE.  That's surprising - I 
#' expected it to be a little worse.  I'll compare this performance with other 
#' models.
#' 


#' ## OLS with Stepwise Selection
#'
model.step <- train(
  x = ols_recipe_7,
  data = d_lin_train80,
  method = "leapSeq",
  tuneGrid = data.frame(nvmax = 50:80),
  metric = "RMSE",
  trControl = model.trControl
)

#' How many predictors did the algorithm use?
print(model.step)

#' 72 predictors.  Here is a plot of the number of predictors that minimize the
#' RMSE.
plot(model.step)

#' Which predictors were both significant and meaningful?  I standardized the 
#' predictors in the model, so I can check this by just comparing 
#' the coefficients.  The coefficient values indicate the effect of a 1 SD 
#' change in the predictor value on the log SalePrice.
#' Here are the predictors with p-values <.05 and estimates > 0.1.
#' 
summary(model.step)$coefficients %>%
  data.frame() %>%
  rownames_to_column(var = "Variable") %>%
  mutate(p = `Pr...t..`) %>%
  mutate(est2 = exp(Estimate)) %>%
  filter(abs(Estimate) > .01 & p < .05 & Variable != "(Intercept)") %>%
  select(Variable, Estimate, p, est2) %>%
  arrange(desc(abs(Estimate))) %>%
  top_n(10)
model.step$results
#' Overall quality (no surprise), followed by basement size (a bit of a 
#' surprise), then second and first floor size.  These all seem good.  Okay, 
#' moving on, how does the model perform against the validation set?
#' 
#' How did the model perform in terms of the within-sample RMSE?
#' 

print(model.step.best <- model.step$bestTune %>% pull())
print(coef(model.step$finalModel, model.step.best))
(perf.step <- postResample(pred = predict(model.step, newdata = d_lin_train20), 
                         obs = d_lin_train20$logSalePrice))

(perf.lm <- postResample(pred = predict(model.lm, newdata = d.valid), 
                         obs = d.valid$logSalePrice))


#' ## PLS
#' ## Regularization Models 
#' ### Ridge

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

#' ### Lasso
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


#' ### Elastic Net
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

#' # Decision Trees 

#' ## Bagged
#' (See discussion on pssing in matrix instead of formula here: https://stats.stackexchange.com/questions/115470/mtry-tuning-given-by-caret-higher-than-the-number-of-predictors)
model.bag = train(
  #  logSalePrice ~ . -Id -SalePrice,
  #  data = d.train80,
  x = d.train80[, -c(1, 2, 88)], 
  y = d.train80[, 88],  # pass in matrix instead of formula
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


#' ## Random Forest
#' ## Boosting


#' # Model Evaluation
#' 
rbind(perf.lm, perf.ridge, perf.lasso, perf.elnet, perf.bag)
models <- list(lm = model.lm,
               ridge = model.ridge, 
               lasso = model.lasso, 
               elnet = model.elnet,
               bag = model.bag)
resamples(models) %>% summary(metric = "RMSE")
bwplot(resamples(models), metric = "RMSE", main = "Model Comparison on Resamples")


#' # Fit Final Model
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

#' Create Submission
preds <- predict(model.final, newdata = d.test)
sub <- data.frame(Id = d.test$Id, SalePrice = exp(preds))
write.csv(sub, file = "./ames_03.csv", row.names = F)
