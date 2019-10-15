#' ---
#' title: "Kaggle - House Prices"
#' subtitle: "Step 2: Data Exploration"
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
library(gridExtra)
library(caret)  # for nearZeroVar()

#' # Load Data
d <- readRDS(file = "./ames_01.RDS")


#' # Univariate Analysis
#' 
#' In this section I will look at data distribution in each variable. For factor
#' variables, I am interested in which are near-zero-variance predictors.  I won't
#' remove them here, but I might remove them in the modeling pre-process step if
#' they cause zero-variance in cv folds (see discussion 
#' [here](https://www.r-bloggers.com/near-zero-variance-predictors-should-we-remove-them/)).
#' `caret` has a nice function to identify near-zero variance predictors.  It defines near-zero variance
#' as a frequency ratio (ratio of the most common value frequency to the second most common 
#' value frequency) >= 19 and a percent of unique values <= 10%. (see [caret package 
#' documentation](https://topepo.github.io/caret/index.html)).
#' (DataCamp course [Machine Learning Toolbox](https://campus.datacamp.com/courses/machine-learning-toolbox/preprocessing-your-data?ex=13))
#' suggests being more aggressive and setting thresholds of frequency ratio >= 2 and percent
#' of unique values <= 20%.

col_is_factor <- lapply(d, function(x) class(x)[1]) %in% c("ordered", "factor")
(nzv <- nearZeroVar(d[, col_is_factor], 
            saveMetrics= TRUE,
            freqCut = 2, uniqueCut = 20))

#' THere are `r = nrow(nzv[nzv$nzv == TRUE])`


myBarPlot <- function(col_name) {
  # So you can pass col_name as a string or symbol
  col_name <- enexpr(col_name)
  if(!is.symbol(col_name)) col_name <- sym(col_name)
  col_name <- enquo(col_name) 
  
  d %>% 
    ggplot(aes(x = !! col_name, y = ..count.., group = 1)) + 
    geom_bar(fill = "cadetblue") +
    geom_text(stat = "count", aes(label = round(..prop.., 2), y = ..count..)) +
    geom_label(stat = "count", aes(label = round(..prop.., 2), y = ..count..)) +
    theme_minimal() +
    theme(legend.position = "none")
}

myBarPlot("BldgType")

skimr::skim_to_wide(d[, col_is_factor])

col_is_numeric <- lapply(d, function(x) class(x)[1]) %in% c("integer", "numeric")


explore_factor <- function(col_name) {
  # So you can pass col_name as a string or symbol
  col_name <- enexpr(col_name)
  if(!is.symbol(col_name)) col_name <- sym(col_name)
  col_name <- enquo(col_name) 
  
  d %>% 
    filter(Set == "Train") %>%
    ggplot(aes(y = SalePrice, x = !! col_name)) + 
    geom_jitter(aes(color = !! col_name), width = 0.2) + 
    geom_boxplot(aes(fill = !! col_name), alpha = 0.5, outlier.shape = NA) + 
    stat_boxplot(geom = "errorbar", width = 0.4) +
    scale_y_continuous(labels = scales::comma) +
    theme_minimal() +
    theme(legend.position = "none")
}


#' ## Add Features
#' I will apply my subject matter knowledge to propose new variables derived from others.  
d <- d %>%
  mutate(TotalSF = TotalBsmtSF + GrLivArea,
         TotalBath = BsmtFullBath + 0.5 * BsmtHalfBath + FullBath + 0.5 * HalfBath,
         AbvGrdSFperRm = GrLivArea / TotRmsAbvGrd,
         logSalePrice = log(SalePrice)
  ) %>% data.frame()


#' Now that the data tranformation tasks are complete, I'll split `d` back into `d.train` and `d.test`.
d.train <- d[d$Set == "train", which(colnames(d) != "Set")]
d.test <- d[d$Set == "test", which(colnames(d) != "Set")]
#d.test$SalePrice


#' ## Handle Outliers

#' Now I'll look at influential outliers.  When building a predictive model, I only need to address 
#' influential outliers for variables that are likely to be important in the final model.  I'll identify 
#' the most likely important variables with a correlation matrix of the numerical predictors.
col_is_numeric <- lapply(d.train, function(x) class(x)[1]) %in% c("integer", "numeric")
names(col_is_numeric) <- names(d.train)
summary(d.train[, col_is_numeric]) %>% print()
(var_num.cor <- cor(d.train[, col_is_numeric]) %>%
    data.frame() %>%
    select(SalePrice) %>%
    data.frame(Variable = rownames(.), Corr = .) %>%
    arrange(desc(SalePrice)))
rm(col_is_numeric)

#' For every variable with correlation >= 0.5 (a moderate correlation), I'll check the variable distribution to see whether any influential outliers exist.
var_num.cor <- var_num.cor[abs(var_num.cor$SalePrice) >= 0.5 & var_num.cor$Variable != "SalePrice",]
df <- data.frame(SalePrice = d.train$SalePrice)
for(i in 1:nrow(var_num.cor)) {
  var <- as.character(var_num.cor[i, "Variable"])
  df$x = unlist(d.train[, "TotalSF"])
  p1 <- ggplot(df, aes(x = x)) + geom_histogram() + labs(x = var)
  p2 <- ggplot(df, aes(x = x, y = SalePrice)) + geom_point() + labs(x = var) + 
    geom_smooth(method = "lm", se = FALSE, na.rm = TRUE)
  grid.arrange(p1, p2, ncol = 2)
} 
rm(var_num.cor)
rm(p1)
rm(p2)
rm(df)
rm(var)
rm(i)

#' I count six potential influential outliers.  They are based on the union of the conditions in the filter condition below.  I can see why the author recommends dropping `GrLivArea` > 4000 - it captures 4 of them.
d.train %>% filter(TotalSF >= 6000 |
                     GrLivArea >= 4000 |
                     TotalBsmtSF >= 6000 | 
                     X1stFlrSF >= 4000 |
                     TotRmsAbvGrd >= 14) %>%
  select(Id, SalePrice, TotalSF, GrLivArea, TotalBsmtSF, X1stFlrSF, TotRmsAbvGrd)

d.train <- d.train %>% filter(TotalSF < 6000 &
                                GrLivArea < 4000 &
                                TotalBsmtSF < 6000 & 
                                X1stFlrSF < 4000 &
                                TotRmsAbvGrd < 14 &
                                LotArea < 100000) %>% 
  data.frame()


#' # Save Work
#' 
#' Save the data as an input to the next step, exploratory data analysis

saveRDS(d.train, file = "./ames_02_train.RDS")
saveRDS(d.test, file = "./ames_02_test.RDS")
