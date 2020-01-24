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


#' My plan is to fit a simple linear regression model, then try several 
#' subsetting and regularization models (stepwise selection, ridge, lasso, 
#' elastic net), and then tree-based models (bagged, random forest, and 
#' boosting).
#' 
#' # Setup
#' 
#' Load the libraries, and the data sets produced in the prior step.
 
#+ echo = FALSE
#+ setup, message = FALSE, warning = FALSE
library(tidyverse)
library(caret)  # for machine learning workflow
#library(broom)
library(e1071)  # for skewness
library(recipes)

d.train <- readRDS(file = "./ames_02_train.RDS")
d.test <- readRDS(file = "./ames_02_test.RDS")


#' # Linear Regression
#' 
#' All linear models are of the form
#' 
#' $$y = \beta X + \epsilon$$
#' 
#' and estimate the $\beta$ coefficients by minimizing the sum of squared 
#' errors, or minimizing a function of the squared errors.
#' 
#' Linear models are relatively easy to interpret, and can be used for 
#' inferential analysis.  A good framework for
#' thinking about the various linear models is how they handle the 
#' bias-variance trade-off.  Ordinary least squares (OLS) estimates 
#' *minimize bias*. Subsetting and regularization models introduce some bias in 
#' order to *reduce variance*.
#'
#' ## OLS
#' 
#' I'll start with OLS regression as a benchmark model.  OLS regression 
#' estimates the model coefficients by minimizing the sum of squared errors,
#'  
#' $$SSE = \sum{(y_i - \hat{y}_i)^2}.$$
#' 
#' The solution is $\hat{\beta} = (X'X)^{-1}X'y$. The OLS model is the best 
#' linear unbiased estimator if the residuals are independent random variables, 
#' normally distributed with mean zero and constant variance.  
#' 
#' I'll need to validate the model assumptions. I'll also be wary of 
#' multicollinearity and near-zero variance (NZV) predictors.  A unique inverse 
#' $(X'X)^{-1}$ only exists when there is no perfect multicollinearity and the 
#' matrix is of full rank, but even when predictors 
#' are only *highly* correlated, the regression coefficient estimations will 
#' still be unstable and can produce a poorly performing model.  
#' 
#' Predictive models are also vulnerable to near-zero variance (NZV) 
#' predictors.  NZV predictors contain little information, so they usually do
#' not add much value to a linear regression.  When cross-validating a model, 
#' NZV predictors can devolve into zero-variance predictors
#' in the individual folds and break the model.  
#'
#' I'll start by addressing NZV.  I'll collapse factor levels so that there
#' are less NZV predictors after they are coded into dummies.  This is 
#' more or less trial and error with the aid of the box-plots and anova post-hoc 
#' testing from the exploratory phase.  The collapsing choices below improved
#' the out-of-sample RMSE.
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
  dplyr::select(-SalePrice, -Condition1, -Condition2, -CondRoad, -CondRail, -CondPos)

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
model.trControl <- trainControl(
  method = "cv",
  # specify the 10 folds instead of n = 10
  index = createFolds(
    d_lin_train80$logSalePrice, 
    k = 10, 
    returnTrain = TRUE
    ),  
  verboseIter = FALSE,  # print results to log?
  # save predictions for optimal tuning params
  savePredictions = "final"
  )  

#' Create a recipe object to contol preprocessing.  Here is the basic recipe.
#' I'll add to it in pieces.
#' 
ols_recipe <- d_lin_train80 %>%
  recipe(logSalePrice ~ .) %>%
  update_role(Id, new_role = "ID")

#' Convert the nominal factor variables to dummy variables.  This balloons the 
#' predictor set up to 236 variables.
#' 
ols_recipe_2 <- ols_recipe %>%
  step_dummy(all_nominal()) 
ols_baked_2 <- ols_recipe_2 %>% prep(d_lin_train80) %>% bake(d_lin_train80)
ols_baked_2 %>% names() %>% length() - 2

#' Remove the NZVs.  I'll use the `nearZeroVar` default thresholds to identify
#' NZVs: ratio of most common to second most common variable >= 95/5, and 
#' percentage of unique values <10%.
#' 
nearZeroVar(ols_baked_2, saveMetrics = TRUE) %>% 
  rownames_to_column(var = "predictor") %>% 
  filter(nzv == TRUE)

#' 93 predictors are NZV.  `step_nzv()` removes them.  
#' 
ols_recipe_3 <- ols_recipe_2 %>%
  step_nzv(all_predictors()) 
ols_baked_3 <- ols_recipe_3 %>% prep(d_lin_train80) %>% bake(d_lin_train80)
ols_baked_3 %>% names() %>% length() - 2

#' That leaves 142 predictors.  Remove predictors that are exact linear 
#' combinations of other predictors.  The first predictor in each list from
#' `findLinearCombos()` is a linear combination of the rest of the list.
#' 
lin_combos <- findLinearCombos(ols_baked_3)
map(lin_combos$linearCombos, function(x) colnames(ols_baked_3)[x])
colnames(ols_baked_3)[lin_combos$remove]

#' 9 predictors are linear combinations of other predictors.  `step_lincomb` 
#' removes them.
#' 
ols_recipe_4 <- ols_recipe_3 %>%
  step_lincomb(all_numeric(), -all_outcomes()) 
ols_baked_4 <- ols_recipe_4 %>% prep(d_lin_train80) %>% bake(d_lin_train80)
ols_baked_4 %>% names() %>% length() - 2

#' That leaves 133 predictors.

#' Remove predictors that have large correlations with the 
#' other predictors.  I'll use the `findCorrelation()` default threshold of 
#' 0.90.
#' 
cor_vars <- findCorrelation(cor(ols_baked_4))
colnames(ols_baked_4)[cor_vars]

#' 11 predictors are highly correlated with other predictors.  `step_corr` 
#' removes them.
#' 
ols_recipe_5 <- ols_recipe_4 %>%
  step_corr(all_numeric(), -all_outcomes(), threshold = .90) 
ols_baked_5 <- ols_recipe_5 %>% prep(d_lin_train80) %>% bake(d_lin_train80)
ols_baked_5 %>% names() %>% length() - 2

#' That leaves 122 predictors. Box-Cox transformations are usually applied to 
#' the response variable, but you can apply it to predictor variables too.  I 
#' don't understand why this makes sense, but it did help the model performance 
#' in this case, so I'll include it.  Here are the predictors (excluding
#' dummies) with significant (>=0.5) skew.
#' 
col_is_num <- map(d_lin_train80, ~ class(.x)[1] %in% c("integer", "numeric")) %>% unlist() 
col_skew <- map(d_lin_train80[, col_is_num], skewness) %>% unlist()
data.frame(var = names(col_skew), skew = col_skew) %>% 
  filter(abs(skew) > 0.5) %>%
  arrange(-abs(skew))

#' Box-Cox will only transform the strictly positive variables.
#' 
boxcox_vars <- preProcess(as.data.frame(ols_baked_5), method = "BoxCox")
boxcox_vars$bc %>% names() %>% setdiff(c("Id", "logSalePrice"))

#' 10 predictors have positive values that can be transformed. `step_BoxCox`
#' transforms them.
#' 
ols_recipe_6 <- ols_recipe_5 %>%
  step_BoxCox(all_numeric(), -all_outcomes())
ols_baked_6 <- ols_recipe_6 %>% prep(d_lin_train80) %>% bake(d_lin_train80)

#' Centering and scaling the data will not affect the model performance, but it
#' provides a useful way to compare the predictor effects on the response.
#' 
ols_recipe_7 <- ols_recipe_6 %>%
  step_center(all_predictors(), -all_nominal()) %>%
  step_scale(all_predictors(), -all_nominal()) 

#' Now train the model.
#' 
#+ model_lm
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

#' The within-sample RMSE was `r round(model.lm$results$RMSE, 4)`.  The 
#' within-sample RMSE is the average
#' of 10 hold-out fold RMSEs, `mean(model.lm$resample$RMSE)`.  Here are 
#' the results of the individual folds.
#'  
#+ fig.height = 3
model.lm$resample %>%
  ggplot(aes(x = Resample, y = RMSE)) +
  geom_col(fill = "cadetblue") +
  theme_minimal() +
  labs(title = "lm model CV RMSE")

#' Here is an observed vs predicted plot with prediction errors > 0.50 flagged.
#' 
model.lm$pred %>% 
  mutate(label = ifelse(abs(obs - pred) > 0.50, rowIndex, "")) %>%
  ggplot(aes(x = pred, y = obs, label = label)) + 
  geom_point(alpha = .2, color = "cadetblue") + 
  geom_abline(slope = 1, intercept = 0, color = "grey40", linetype = "longdash") +
  geom_abline(slope = 1, intercept = -.50, color = "grey40", linetype = "dotted") +
  geom_abline(slope = 1, intercept = +.50, color = "grey40", linetype = "dotted") +
  geom_text(size = 2, color = "cadetblue", position = position_jitter(width = .3, height = .3)) +
  expand_limits(x = c(9, 15), y = c(9, 15)) +
  labs(title = "lm Observed vs Predicted")

#' Looks like the model makes the worst mistakes predicting the values of 
#' inexpensive houses, usually overestimating their sale price.  Does the model 
#' conform to the linear model assumption that residuals are 
#' independent random variables normally distributed with mean zero and 
#' constant variance?  A standardized residuals vs fits plot should vary around e=0
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
  geom_abline(slope = 0, intercept = 2, color = "grey40", linetype = "dotted") +
  geom_abline(slope = 0, intercept = -2, color = "grey40", linetype = "dotted") +
  geom_text(size = 2, position = position_jitter(width = .3, height = .3), color = "cadetblue") +
  labs(title = "lm Standardized Residuals vs Predicted")

#' A normal probability plot compares the theoretical percentiles of the normal 
#' distribution versus the observed sample percentiles. It should be 
#' approximately linear.  
stdres <- model.lm$pred %>% 
  mutate(res = pred-obs,
         stdres = res / sd(res)) %>%
  pull(stdres)
qqnorm(stdres, main = "lm Normal Q-Q Plot")
qqline(stdres)

#' This one does not look so good.  The distribution is heavy tailed. There 
#' are many extreme positive and negative residuals.  The Anderson-Darling 
#' normality test p-value is the probability of calculating the test statistic 
#' if the distribution is normal.  This one is not normal.
model.lm$pred %>% 
  mutate(res = pred-obs) %>%
  pull(res) %>% 
  nortest::ad.test()

#' I'm not sure how to proceed here - use a different transformation for the
#' response variable?  I'm going to move on.  Which predictors were both 
#' significant and meaningful?  I standardized the 
#' predictors in the model, so I can compare 
#' the coefficients.  The coefficient values indicate the effect of a 1 SD 
#' change in the predictor value on the log SalePrice.
#' Here are the predictors with p-values <.05 and coefficient estimates > 0.01.
#' 
summary(model.lm)$coefficients %>%
  data.frame() %>%
  rownames_to_column(var = "Variable") %>%
  mutate(p = `Pr...t..`) %>%
  filter(abs(Estimate) > .01 & p < .05 & Variable != "(Intercept)") %>%
  dplyr::select(Variable, Estimate, p) %>%
  arrange(desc(abs(Estimate))) %>%
  top_n(10, wt = Estimate)

#' Overall quality (no surprise), followed by basement size (a bit of a 
#' surprise), then second and first floor size.  These all seem good. Another 
#' way to evaluate the model is with "variable importance".  For 
#' linear models, that is defined by the absolute value of the t-statistic.
#' 
varImp(model.lm)

#' Moving on, how does the model perform against the out-of-sample set?
#' 
(perf.lm <- postResample(pred = predict(model.lm, newdata = d_lin_train20), 
                         obs = d_lin_train20$logSalePrice))

#' It's slightly better than the within-sample RMSE.  That's surprising - I 
#' expected it to be a little worse.  I'll compare this performance with other 
#' models.
#' 

#' What if I treat `OverallQual` and `OverallCon` as numeric variables?
#' 
d_lin_train80b <- d_lin_train80 %>%
  mutate(
    OverallQual = as.numeric(OverallQual),
    OverallCond = as.numeric(OverallCond)
  )
d_lin_train20b <- d_lin_train20 %>%
  mutate(
    OverallQual = as.numeric(OverallQual),
    OverallCond = as.numeric(OverallCond)
  )
model.lmb <- train(
  x = ols_recipe_7,
  data = d_lin_train80b,
  method = "lm",
  metric = "RMSE",
  trControl = model.trControl
)
(perf.lmb <- postResample(pred = predict(model.lmb, newdata = d_lin_train20b), 
                         obs = d_lin_train20b$logSalePrice))

#' The RMSE improved by 0.0022.

#' ## Stepwise Selection
#'
#' The more predictors that are included in the model, the smaller the 
#' within-sample RMSE. This does not guarantee better predictive performance 
#' out-of-sample though, because the model coefficients are less stable when 
#' they are correlated with each other.  Stepwise selection iteratively adds 
#' and removes predictors in the model to 
#' find the subset of predictors that minimize the model prediction error.
#' 
#+ model_step, cache=TRUE
model.lmStepAIC <- train(
  x = ols_recipe_7,
  data = d_lin_train80,
  method = "lmStepAIC", 
  trace = FALSE,
  metric = "RMSE",
  trControl = model.trControl
)

#' What did I get for a model?
#' 
summary(model.lmStepAIC)

#' How did the model perform in terms 
#' of the within-sample RMSE?
#' 
print(model.lmStepAIC)

#' The within-sample RMSE was `r round(model.lmStepAIC$results$RMSE, 4)`.  This is
#' actually a skosh *better* than the OLS model.  Here are 
#' the results of the individual folds.
#'  
#+ fig.height = 3
model.lmStepAIC$resample %>%
  ggplot(aes(x = Resample, y = RMSE)) +
  geom_col(fill = "cadetblue") +
  theme_minimal() +
  labs(title = "stepAIC model CV RMSE")

#' Fold01 and Fold03 were the worst again.  Here is the observed vs predicted 
#' plot with prediction errors > 0.50 flagged.
#' 
model.lmStepAIC$pred %>% 
  mutate(label = ifelse(abs(obs - pred) > 0.50, rowIndex, "")) %>%
  ggplot(aes(x = pred, y = obs, label = label)) + 
  geom_point(alpha = .2, color = "cadetblue") + 
  geom_abline(slope = 1, intercept = 0, color = "grey40", linetype = "longdash") +
  geom_abline(slope = 1, intercept = -.50, color = "grey40", linetype = "dotted") +
  geom_abline(slope = 1, intercept = +.50, color = "grey40", linetype = "dotted") +
  geom_text(size = 2, color = "cadetblue", position = position_jitter(width = .3, height = .3)) +
  expand_limits(x = c(9, 15), y = c(9, 15)) +
  labs(title = "stepAIC Observed vs Predicted")

#' Does the model conform to the linear model assumption that residuals are 
#' independent random variables normally distributed with mean zero and 
#' constant variance?  Here is the standardized residuals vs fits plot.
#' 
model.lmStepAIC$pred %>% 
  mutate(res = pred-obs,
         stdres = res / sd(res),
         label = ifelse(abs(stdres) > 2, rowIndex, "")) %>%
  ggplot(aes(x = pred, y = stdres, label = label)) + 
  geom_point(alpha = .2, color = "cadetblue") + 
  geom_abline(slope = 0, intercept = 2, color = "grey40", linetype = "dotted") +
  geom_abline(slope = 0, intercept = -2, color = "grey40", linetype = "dotted") +
  geom_text(size = 2, position = position_jitter(width = .3, height = .3), color = "cadetblue") +
  #  expand_limits(x = c(7.5, 15), y = c(7.5, 15)) +
  labs(title = "stepAIC Standardized Residuals vs Predicted")

#' Here is the normal probability plot.  
stdres <- model.lmStepAIC$pred %>% 
  mutate(res = pred-obs,
         stdres = res / sd(res)) %>%
  pull(stdres)
qqnorm(stdres, main = "stepAIC Normal Q-Q Plot")
qqline(stdres)

#' Not so good.  The distribution is heavy tailed again.  The Anderson-Darling 
#' normality test agrees.
model.lmStepAIC$pred %>% 
  mutate(res = pred-obs) %>%
  pull(res) %>% 
  nortest::ad.test()

#' Which predictors were both significant and meaningful?  Here are the 
#' predictors with p-values <.05 and estimates > 0.1.
#' 
summary(model.lmStepAIC)$coefficients %>%
  data.frame() %>%
  rownames_to_column(var = "Variable") %>%
  mutate(p = `Pr...t..`) %>%
  filter(abs(Estimate) > .01 & p < .05 & Variable != "(Intercept)") %>%
  dplyr::select(Variable, Estimate, p) %>%
  arrange(desc(abs(Estimate))) %>%
  top_n(10, wt = Estimate)

#' The top 10 predictors from `stepAIC` are nearly identical to those in `lm`, 
#' except for the introduction of  `GrLivArea`. How about the variable importance?
varImp(model.lmStepAIC)

#' Similar to my "significant and meaningful" again.  How does the model perform 
#' against the out-of-sample set?
#' 
perf.step <- postResample(
  pred = predict(model.lmStepAIC, newdata = d_lin_train20), 
  obs = d_lin_train20$logSalePrice
  )
perf.step

#' Reassuringly, it's slightly worse than the within-sample RMSE.  It's also about
#' the same as the all-in OLS model I fit previously.  And this model has only
#' `r length(model.lmStepAIC$finalModel$coefficients) - 1` predictors compared to 
#' `r length(model.lm$finalModel$coefficients) - 1` in the original model!
#' 
#'
#' Again, what if I treat `OverallQual` and `OverallCon` as numeric variables?
#' 

model.lmStepAICb <- train(
  x = ols_recipe_7,
  data = d_lin_train80b,
  method = "lmStepAIC", 
  trace = FALSE,
  metric = "RMSE",
  trControl = model.trControl
)
perf.stepb <- postResample(
  pred = predict(model.lmStepAICb, newdata = d_lin_train20b), 
  obs = d_lin_train20b$logSalePrice
)
perf.stepb

#' Treating `OverallQual` and `OverallCon` as numeric variables improved the 
#' RMSE by 0.026.

#' ## Regularization Models 
#' 
#' Regularization is an approach that reduces variance at the cost of 
#' introducing some bias. Stepwise selection balances this tradeoff by 
#' eliminating variables, but this throws away information. Regularization 
#' keeps all the predictors, but reduces coefficient values to reduce bias.
#' 
#' The collapsed factors that worked well for the simple OLS and stepwise 
#' selection models did not work so well here, so I'll use the original data.
#' 
d.train80 <- d.train[partition, !(colnames(d.train) %in% c("Id", "SalePrice"))]
d.train20 <- d.train[-partition, !(colnames(d.train) %in% c("Id", "SalePrice"))]

#' ### Ridge
#'
#' The Ridge approach estimates the linear model coefficients by minimizing 
#' an augmented loss fuction which includes a term, $\lambda$, that penalizes 
#' the magnitude of the coefficient estimates,
#' 
#' $$L_{ridge} = ||y - X \hat{\beta}||^2 + \lambda||\hat\beta||^2.$$
#' 
#' The resulting estimate for the coefficients is 
#' 
#' $$\hat{\beta} = (X'X + \lambda I)^{-1}(X'Y).$$
#' 
#' As $\lambda \rightarrow 0$, ridge regression approaches OLS.  The bias and 
#' variance for the ridge estimator is
#' 
#' $$Bias(\hat{\beta}_{ridge}) = -\lambda (X'X + \lambda I)^{-1} \beta$$
#' 
#' $$Var(\hat{\beta}_{OLS}) = \sigma^2(X'X + \lambda I)^{-1}X'X(X'X + \lambda I)^{-1}$$
#' 
#' The estimator bias increases with $\lambda$ and the estimator variance 
#' decreases with $\lambda$.  The optimal level for $\lambda$ that balances the 
#' tradeoff is the one that minimizes some criterion with cross-validation, 
#' usually the root mean squared error (RMSE) (but you can also use the Akaike 
#' or Bayesian Information Criterion (AIC or BIC), or Rsquared).
#' 
#' Perform cross-validation to optimize $\lambda$, specifying `alpha = 0` in a 
#' tuning grid for ridge regression.
#' 

# pen_recipe <-
#   recipe(logSalePrice ~ ., data = d.train80) %>%
#   step_zv(all_predictors()) %>%
#   step_center(all_predictors(), -all_nominal()) %>%
#   step_scale(all_predictors(), -all_nominal()) %>%
#   step_dummy(all_predictors(), -all_numeric()) 
  
#+ model_ridge
model.ridge <- train(
  logSalePrice ~ .,
#  x = pen_recipe,
  data = d.train80,
  method = "glmnet",
#  standardize = FALSE,
  metric = "RMSE", 
  preProcess = c("zv", "center", "scale"),  
  # tweak a tuning grid to zero in on best hyperparameter settings.
  tuneGrid = expand.grid(
    .alpha = 0,  # alpha = 0 for ridge regression
    .lambda = seq(0, 0.5, length = 100)),
  trControl = model.trControl
)

#' Here are the tuning results.
#' 
model.ridge$bestTune

#' Here is a plot of the tuning process.  I tweaked the tuning grid to hone in
#' on the minimum lambda.
#' 
plot(model.ridge, main = "Ridge Regression Parameter Tuning", xlab = "lambda")

#' What were the most important variables?
#'
varImp(model.ridge)

#' How does the model perform against the out-of-sample set?
#' 
(perf.ridge <- postResample(
  pred = predict(model.ridge, newdata = d.train20), 
  obs = d.train20$logSalePrice
  ))

#' Not bad.  A new front-runner.
#'
#' What if I treat `OverallQual` and `OverallCon` as numeric variables?
#' 
d.train80b <- d.train80 %>%
  mutate(
    OverallQual = as.numeric(OverallQual),
    OverallCond = as.numeric(OverallCond)
  )
d.train20b <- d.train20 %>%
  mutate(
    OverallQual = as.numeric(OverallQual),
    OverallCond = as.numeric(OverallCond)
  )
model.ridgeb <- train(
  logSalePrice ~ .,
  #  x = pen_recipe,
  data = d.train80b,
  method = "glmnet",
  #  standardize = FALSE,
  metric = "RMSE", 
  preProcess = c("zv", "center", "scale"),  
  # tweak a tuning grid to zero in on best hyperparameter settings.
  tuneGrid = expand.grid(
    .alpha = 0,  # alpha = 0 for ridge regression
    .lambda = seq(0, 0.5, length = 100)),
  trControl = model.trControl
)
(perf.ridgeb <- postResample(
  pred = predict(model.ridgeb, newdata = d.train20b), 
  obs = d.train20b$logSalePrice
))

#' The RMSE *increased* very slightly (0.0004).
#' 
#' ### Lasso
#' 
#' Lasso stands for “least absolute shrinkage and selection operator”.  Like 
#' ridge, lasso adds a penalty for coefficients, but instead of penalizing the 
#' sum of squared coefficients (L2 penalty), lasso penalizes the sum of absolute 
#' values (L1 penalty). As a result, for high values of $\lambda$, coefficients 
#' can be zeroed under lasso.
#' 
#' The loss fuction for lasso is
#' 
#' $$L_{ridge} = ||y - X \hat{\beta}||^2 + \lambda||\hat\beta||.$$
#' 
  
#+ model_lasso
model.lasso <- train(
  logSalePrice ~ .,
  data = d.train80,
  method = "glmnet",
  metric = "RMSE", 
  preProcess = c("zv", "center", "scale"),  
  tuneGrid = expand.grid(
    .alpha = 1,  # alpha = 1 for lasso regression
    .lambda = seq(0, 0.025, length = 100)),
  trControl = model.trControl
)

#' Here are the tuning results.
#' 
model.lasso$bestTune

#' Here is a plot of the tuning process for lambda.
#' 
plot(model.lasso, main = "Lasso Regression Parameter Tuning", xlab = "lambda")

#' What were the most important variables?
#'
varImp(model.lasso)

#' How does the model perform against the out-of-sample set?
#' 
perf.lasso <- postResample(
  pred = predict(model.lasso, newdata = d.train20),
  obs = d.train20$logSalePrice)
perf.lasso

#' Good.  A new front-runner again.  Now try treating `OverallQual` and `OverallCond` as numerics.
#' 
model.lassob <- train(
  logSalePrice ~ .,
  data = d.train80b,
  method = "glmnet",
  metric = "RMSE", 
  preProcess = c("zv", "center", "scale"),  
  tuneGrid = expand.grid(
    .alpha = 1,  # alpha = 1 for lasso regression
    .lambda = seq(0, 0.025, length = 100)),
  trControl = model.trControl
)
perf.lassob <- postResample(
  pred = predict(model.lassob, newdata = d.train20b),
  obs = d.train20b$logSalePrice)
perf.lassob

#' Again, a small increase in the RMSE (0.0004).
#' 
#'
#' ### Elastic Net
#' 
#' Elastic Net combines the penalties of ridge and lasso to get the best of 
#' both worlds. The loss fuction for elastic net is
#' 
#' $$L_{enet} = \frac{||y - X \hat{\beta}||^2}{2n} + \lambda \frac{1 - \alpha}{2}||\hat\beta||^2 + \lambda \alpha||\hat\beta||.$$
#' 
#' In this loss function, new parameter $\alpha$ is a "mixing" parameter that 
#' balances the two approaches.  You can see that if $\alpha$ is zero, you are 
#' back to ridge regression, and if $\alpha$ is one, you are back to lasso.

#+ model_elnet
model.elnet <- train(
  logSalePrice ~ .,
  data = d.train80,
  method = "glmnet",
  metric = "RMSE", 
  preProcess = c("zv", "center", "scale"),  
  tuneGrid = expand.grid(
    .alpha = seq(0, 0.25, length = 10),  # alpha varies for elastic net regression
    .lambda = seq(0, 0.05, length = 10)),
  trControl = model.trControl
)

#' Here are the tuning results.
#' 
model.elnet$bestTune

#' Here is a plot of the tuning process.  There are two hyperparameters this 
#' time: alpha and lambda.
#' 
plot(model.elnet, main = "ElNet Regression Parameter Tuning", xlab = "alpha")

#' What were the most important variables?
#'
varImp(model.elnet)

#' How does the model perform against the out-of-sample set?
#' 
perf.elnet <- postResample(
  pred = predict(model.elnet, newdata = d.train20), 
  obs = d.train20$logSalePrice)
perf.elnet

#' Life just gets better and better.  Now try treating `OverallQual` and `OverallCond` as numerics.
#' 
model.elnetb <- train(
  logSalePrice ~ .,
  data = d.train80b,
  method = "glmnet",
  metric = "RMSE", 
  preProcess = c("zv", "center", "scale"),  
  tuneGrid = expand.grid(
    .alpha = seq(0, 0.25, length = 10),  # alpha varies for elastic net regression
    .lambda = seq(0, 0.05, length = 10)),
  trControl = model.trControl
)
perf.elnetb <- postResample(
  pred = predict(model.elnetb, newdata = d.train20b), 
  obs = d.train20b$logSalePrice)
perf.elnetb

#' Again, a small increase in the RMSE (0.0006).
#' 
#'
#' # Decision Trees 
#'
#' Classification and regression trees (CART) segment the predictor space into 
#' non-overlapping regions, the nodes of the tree.  Each node is described by 
#' a set of rules which are then used to predict new responses. The predicted 
#' value $\hat{y}$ for each node is the mode mean.
#' 
#'  CART recursively partitions the predictor space, starting with all the 
#'  observations in a single node. It splits this node using the best predictor 
#'  variable and cutpoint so that the responses within each subtree are as 
#'  homogenous as possible, and repeats the splitting process for each of the 
#'  child nodes until a stopping criterion is satisfied.  
#'  
#'  CART uses a "greedy" approach to splitting.  For each variable it evaluates 
#'  several split points according to a cost function.  The variable and split 
#'  point that minimizes the cost function is the split.  There are many 
#'  possible cost functions.  For a regression tree, the cost function is the 
#'  sum of squared errors, $\sum(y - \hat{y})^2$.  
#'  
#'  CART uses a minimum count stopping criteria.  If the next split results in 
#'  a node with less than some minimum, CART rejects the split and the current 
#'  node is taken as a final leaf node.
#'  
#'  The resulting tree likely over-fits the training data, so CART "prunes" leaves 
#'  from the tree. The most common pruning method is cost-complexity pruning. 
#'  Cost-complexity pruning minimizes the cost complexity: $CC(T) = R(T) + cp|T|$ 
#'  where $|T|$ is the tree size (complexity), $R(T)$ is the cost (ESS, gini, 
#'  entropy, etc.), and $cp$ is the complexity parameter. CART finds the tree 
#'  with the lowest $CC$ for many $cp$ values, then chooses the tree with the 
#'  lowest $CC$.  Pruning is performed either with the validation dataset, 
#'  or k-fold cross validation.
#'  
#'  Decision trees have limitations.  They only provide course-grain 
#'  predictions (number of leaves) vs continuous predictions in a linear model, 
#'  and do not express truly linear relationships well.
#'  
#' ## Bagged
#' 
#' Bootstrap aggregation, or *bagging*, is a general-purpose procedure for 
#' reducing the variance of a statistical learning method.  The algorithm 
#' constructs *B* regression trees using *B* bootstrapped training sets, and 
#' averages the resulting predictions. These trees are grown deep, and are not 
#' pruned. Hence each individual tree has high variance, but low bias. 
#' Averaging these *B* trees reduces the variance.  Set *B* sufficiently large 
#' that the error has settled down.
#' 
#' To test the model accuracy, the out-of-bag observations are predicted from 
#' the models that do not use them.  If *B/3* of observations are in-bag, 
#' there are *B/3* predictions per observation.  These predictions are 
#' averaged for the test prediction.  
#' 
#' The downside to bagging is that it improves accuracy at the expense of 
#' interpretability.  There is no longer a single tree to interpret, so it is 
#' no longer clear which variables are more important than others.  
#' 
#' The model below uses the `x =`, `y = ` formulation instead of `y ~ x`. (See 
#' the discussion on passing in matrix instead of formula here: 
#' https://stats.stackexchange.com/questions/115470/mtry-tuning-given-by-caret-higher-than-the-number-of-predictors).
#'  
#+ model_bag, cache=TRUE
model.bag <- train(
  # pass in matrix instead of formula
  x = d.train80[, -1], 
  y = d.train80[, 1],  
  method = "ranger",
  metric = "RMSE", 
  preProcess = c("zv"),  
  tuneGrid = expand.grid(
    .mtry = length(d.train80[, -c(1:3)]),  # mtry = p for bagging
    .splitrule = c("variance", "extratrees", "maxstat"),
    .min.node.size = seq(from = 1, to = 10)
    ),
  trControl = model.trControl
)

#' Here are the tuning results.
#' 
model.bag$bestTune

#' Here is a plot of the tuning process.  `mtry` is the number of variables
#' to split at each node.  Bagging uses *all* variables, so `mtry = p`.
#' `splitrule` is the statistic to optimize in the subtrees. `min.node.size` is
#' the minimum node size and defaults to 5 for regression trees.
#' 
plot(model.bag, main = "Bagging Parameter Tuning")

#' How does the model perform against the out-of-sample set?
#' 
perf.bag <- postResample(
  pred = predict(model.bag, newdata = d.train20), 
  obs = d.train20$logSalePrice
  )
perf.bag

#' This is the worst model so far.  Does treating `OverallQual` and 
#' `OverallCond` as numerics help?
model.bagb <- train(
  # pass in matrix instead of formula
  x = d.train80b[, -1], 
  y = d.train80b[, 1],  
  method = "ranger",
  metric = "RMSE", 
  preProcess = c("zv"),  
  tuneGrid = expand.grid(
    .mtry = length(d.train80b[, -c(1:3)]),  # mtry = p for bagging
    .splitrule = c("variance", "extratrees", "maxstat"),
    .min.node.size = seq(from = 1, to = 10)
  ),
  trControl = model.trControl
)
perf.bagb <- postResample(
  pred = predict(model.bagb, newdata = d.train20b), 
  obs = d.train20b$logSalePrice
)
perf.bagb

#' Just about the same performance. 
#' 
#' 
#' ## Random Forest
#' 
#' Random forests improve bagged trees by way of a small tweak that 
#' de-correlates the trees.  As in bagging, the algorithm builds a number of 
#' decision trees on bootstrapped training samples. But when building these 
#' decision trees, each time a split in a tree is considered, a random sample 
#' of *mtry* predictors is chosen as split candidates from the full set 
#' of *p* predictors.  A fresh sample of *mtry* predictors is taken at each 
#' split.  Typically $mtry \sim \sqrt{B}$.  Bagged trees are thus a special 
#' case of random forests where *mtry = p*.
#' 
#+ model_for, cache=TRUE
model.for <- train(
  # pass in matrix instead of formula
  x = d.train80[, -1], 
  y = d.train80[, 1],  
  method = "ranger",
  metric = "RMSE", 
  preProcess = c("zv"),  
  tuneGrid = expand.grid(
    .mtry = seq(from = 50, to = 94, by = 5),  # mtry varies for random forest
    .splitrule = c("extratrees"),  # optimized out variance, maxstat
    .min.node.size = seq(from = 4, to = 5, by = 1)
  ),
  trControl = model.trControl
)

#' Here are the tuning results.
#' 
model.for$bestTune

#' Here is a plot of the tuning process for `mtry`, `splitrule`, and 
#' `min.node.size`.
#' 
plot(model.for, main = "Random Forest Parameter Tuning")

#' How does the model perform against the out-of-sample set?
#' 
perf.for <- postResample(
  pred = predict(model.for, newdata = d.train20),
  obs = d.train20$logSalePrice
  )
perf.for

#' A little better than bagging.  And with the numerice `OveralCond` and `OverallQual`?
#'  
model.forb <- train(
  # pass in matrix instead of formula
  x = d.train80b[, -1], 
  y = d.train80b[, 1],  
  method = "ranger",
  metric = "RMSE", 
  preProcess = c("zv"),  
  tuneGrid = expand.grid(
    .mtry = seq(from = 50, to = 94, by = 5),  # mtry varies for random forest
    .splitrule = c("extratrees"),  # optimized out variance, maxstat
    .min.node.size = seq(from = 4, to = 5, by = 1)
  ),
  trControl = model.trControl
)
perf.forb <- postResample(
  pred = predict(model.forb, newdata = d.train20),
  obs = d.train20b$logSalePrice
)
perf.forb

#' The numerics were a smidge better.
#' 
#' ## Boosting
#' 
#' Boosting is a method to improve (boost) the week learners sequentially and 
#' increase the model accuracy with a combined model. The regression trees are 
#' addative, so that the successive models can be added together to correct 
#' the residuals in the earlier models.  There are several 
#' boosting algorithms.  One of the earliest was AdaBoost (adaptive boost).  A 
#' more recent innovation is gradient boosting.  
#' 
#' The gradient boosting algorithm fits a shallow tree $T_1$ to the data, 
#' $M_1 = T_1$.  Then it fits a tree $T_2$ to the residuals and adds a weighted 
#' sum of the tree to the original tree as $M_2 = M_1 + \gamma T_2$.  For 
#' regularized boosting, include a learning rate factor $\eta \in (0..1)$, 
#' $M_2 = M_1 + \eta \gamma T_2$.  A larger $\eta$ produces faster learning, 
#' but risks overfitting.  The process repeats until the residuals are small 
#' enough, or until it reaches the maximum iterations.  Because overfitting 
#' is a risk, use cross-validation to select the appropriate number of trees 
#' (the number of trees producing the lowest RMSE).
#' 
#' It is common to constrain the weak learners by setting maximum tree 
#' size parameters.  
#' 
#' **Tree Constraints**.  In general the more constrained the tree, the more 
#' trees need to be grown.  Parameters to optimize include number of trees, 
#' tree depth, number of nodes, minimum observations per split, and minimum 
#' improvement to loss.
#' 
#' **Learning Rate**.  Each successive tree can be weighted to slow down the 
#' learning rate.  Decreasing the learning rate increases the number of 
#' required trees.  Common growth rates are 0.1 to 0.3.

#+ model_gbm, cache=TRUE
model.gbm <- train(
  # pass in matrix instead of formula
  x = d.train80[, -1], 
  y = d.train80[, 1],  
  method = "gbm",  # gradient boosting
  metric = "RMSE", 
  distribution = "gaussian",
  verbose = FALSE,
  preProcess = c("zv"), 
  tuneGrid = expand.grid(
    .n.trees = seq(from = 100, to = 400, by = 20),
    .interaction.depth = c(3, 4, 5),
    .shrinkage = c(0.025, 0.050, 0.075),  # aka, learning rate
    .n.minobsinnode = c(2, 3, 4)
  ),
  trControl = model.trControl
)

#' Here are the tuning results.
#' 
model.gbm$bestTune

#' Here is a plot of the tuning process.  I tweaked the tuning grid to hone in
#' on the minimum n.trees, interaction.depth, shrinkage, and n.minobsinnode.
#' 
plot(model.gbm, main = "Gradient Boosting Parameter Tuning")

#' How does the model perform against the out-of-sample set?
#' 
perf.gbm <- postResample(
  pred = predict(model.gbm, newdata = d.train20),
  obs = d.train20$logSalePrice
)
perf.gbm

#' And now with numeric `OverallQual` and `OverallCond`.
#'
model.gbmb <- train(
  # pass in matrix instead of formula
  x = d.train80b[, -1], 
  y = d.train80b[, 1],  
  method = "gbm",  # gradient boosting
  metric = "RMSE", 
  distribution = "gaussian",
  verbose = FALSE,
  preProcess = c("zv"), 
  tuneGrid = expand.grid(
    .n.trees = seq(from = 100, to = 400, by = 20),
    .interaction.depth = c(3, 4, 5),
    .shrinkage = c(0.025, 0.050, 0.075),  # aka, learning rate
    .n.minobsinnode = c(2, 3, 4)
  ),
  trControl = model.trControl
)

perf.gbmb <- postResample(
  pred = predict(model.gbmb, newdata = d.train20b),
  obs = d.train20b$logSalePrice
)
perf.gbmb

#' RMSE improved from 0.1182 to 0.1145.
#' 

#' # Nonlinear Regression
#' 
#' ## MARS
#' 
#' Multivariate adaptive regression splines (MARS) is a non-parametric 
#' algorithm that creates a piecewise linear model to capture nonlinearities 
#' and interactions effects. The resulting model is a weighted sum of *basis* 
#' functions $B_i(X)$:
#' 
#' $$\hat{y} = \sum_{i=1}^{k}{w_iB_i(x)}$$
#' 
#' The basis functions are either a constant (for the intercept), a *hinge* 
#' function of the form $\max(0, x - x_0)$ or $\max(0, x_0 - x)$ (a more 
#' concise representation is $[\pm(x - x_0)]_+$), or products of two or more 
#' hinge functions (for interactions).  MARS automatically selects which 
#' predictors to use and what predictor values to serve as the *knots* of the 
#' hinge functions.
#' 
#' MARS builds a model in two phases: the forward pass and the backward pass, 
#' similar to growing and pruning of tree models. MARS starts with a model 
#' consisting of just the intercept term equaling the mean of the response 
#' values.  It then asseses every predictor to find a basis function pair 
#' consisting of opposing sides of a mirrored hinge function which produces 
#' the maximum improvement in the model error.  MARS repeats the process until 
#' either it reaches a predefined limit of terms or the error improvement 
#' reaches a predefined limit.  MARS generalizes the model by removing terms 
#' according to the generalized cross validation (GCV) criterion.  GCV is a 
#' form of regularization: it trades off goodness-of-fit against model 
#' complexity. 
#' 
#' The `earth::earth()` function ([documentation](https://www.rdocumentation.org/packages/earth/versions/5.1.2/topics/earth)) 
#' performs the MARS algorithm.  The caret implementation tunes two 
#' parameters: `nprune` and `degree`.  `nprune` is the maximum number of terms 
#' in the pruned model.  `degree` is the maximum degree of interaction 
#' (default is 1 (no interactions)).  However, there are other hyperparameters 
#' in the model that may improve performance, including `minspan` which 
#' regulates the number of knots in the predictors.
#' 
#+ model_mars, cache=TRUE
mars_recipe <- 
  recipe(logSalePrice ~ ., data = d.train80)  

model.mars <- train(
  mars_recipe,
  data = d.train80,
  method = "earth",
  metric = "RMSE",
  minspan = -15,
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(
    degree = 1, 
    nprune = seq(2, 100, length.out = 10) %>% floor()
  )
)

#' Here are the tuning results.
#' 
model.mars$bestTune

#' Here is a plot of the tuning process.  I tweaked the tuning grid to hone in
#' on the best degree, nprune, and minspan.
#' 
plot(model.mars, main = "MARS Parameter Tuning")

#' How does the model perform against the out-of-sample set?
#' 
(perf.mars <- postResample(
  pred = predict(model.mars, newdata = d.train20),
  obs = d.train20$logSalePrice
))

#' Better than the tree models, but not as good as any of the penalization 
#' models.  One more try with numeric `OverallQual` and `OverallCond`.
#' 
model.marsb <- train(
  mars_recipe,
  data = d.train80b,
  method = "earth",
  metric = "RMSE",
  minspan = -15,
  trControl = trainControl(method = "cv", number = 10),
  tuneGrid = expand.grid(
    degree = 1, 
    nprune = seq(2, 100, length.out = 10) %>% floor()
  )
)
(perf.marsb <- postResample(
  pred = predict(model.marsb, newdata = d.train20b),
  obs = d.train20b$logSalePrice
))
#' A good improvement again.

#' 
#' 
#' # Model Evaluation
#' 
#' Here is a summary of the fitted model performances on the within-sample 
#' RMSE.
#' 
models <- list(
  lm = model.lm,
  lmb = model.lmb,
  step = model.lmStepAIC, 
  stepb = model.lmStepAICb, 
  ridge = model.ridge, 
  ridgeb = model.ridgeb, 
  lasso = model.lasso, 
  lassob = model.lassob, 
  elnet = model.elnet,
  elnetb = model.elnetb,
  bag = model.bag,
  bagb = model.bagb,
  forestb = model.forb,
  forest = model.for,
  gbm = model.gbm,
  gbmb = model.gbmb,
  mars = model.mars,
  marsb = model.marsb
  )
resamples(models) %>% summary(metric = "RMSE")
bwplot(resamples(models), metric = "RMSE", main = "Model Comparison on Resamples")

#' And here is a summary of the fitted model performances on the out-of-sample
#' RMSE.
#' 
rbind(
  perf.lm, 
  perf.lmb, 
  perf.step, 
  perf.stepb, 
  perf.ridge, 
  perf.ridgeb, 
  perf.lasso, 
  perf.lassob, 
  perf.elnet, 
  perf.elnetb, 
  perf.bag,
  perf.bagb,
  perf.for,
  perf.forb,
  perf.gbm,
  perf.gbmb,
  perf.mars,
  perf.marsb
) %>% data.frame() %>% rownames_to_column(var = "Model") %>% arrange(-RMSE)

#' The big winner is... elastic net. 
#' Congratulations, elastic net!
#' 
#' 
#' # Fit Final Model
#' 
#' Elastic net was the winner, so I'll use it to fit the final model for the 
#' competition submission using the full training data set.
#' 
#+ model_final
model.final <- train(
  logSalePrice ~ .,
  data = d.train[, !(colnames(d.train) %in% c("Id", "SalePrice"))],
  method = "glmnet",
  metric = "RMSE", 
  preProcess = c("zv"),  
  tuneGrid = expand.grid(
    .alpha = model.elnet$bestTune$alpha,
    .lambda = model.elnet$bestTune$lambda
  ),
  trControl = model.trControl
)

model.final

#' # Create Submission File
#' 
#' The submission file is a csv file of Id and SalePrice.
#' 
preds <- predict.train(model.final, newdata = d.test)
sub <- data.frame(Id = d.test$Id, SalePrice = exp(preds))
write.csv(sub, file = "./ames_03.csv", row.names = FALSE)

