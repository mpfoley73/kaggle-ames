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


#' # Setup

#+ echo = FALSE
#+ setup, message = FALSE, warning = FALSE
library(tidyverse)
library(gridExtra)
library(caret)  # for nearZeroVar()
library(e1071)  # for skewness()
library(broom)  # for tidy()
library(agricolae)  # for HSD.test()
library(corrplot)

d <- readRDS(file = "./ames_01.RDS")


#' # Univariate Analysis
#' 
#' In this section I will look at data distributions. For factor
#' variables, I am interested in which are near-zero-variance predictors.  For quantitative 
#' variables, I am looking for significant skew.

col_is_ord <- map(d, ~ class(.x)[1] == "ordered") %>% unlist()

col_is_nom <- map(d, ~ class(.x)[1] == "factor") %>% unlist()

col_is_num <- map(d, ~ class(.x)[1] %in% c("integer", "numeric")) %>% unlist() 
col_is_num <- col_is_num == TRUE & !(names(col_is_num) %in% c("Id", "SalePrice"))

col_is_oth <- colnames(d) %in% colnames(d)[!(col_is_ord | col_is_nom | col_is_num)]
names(col_is_oth) <- colnames(d)

assertthat::are_equal(
  sum(col_is_ord + col_is_nom + col_is_num + col_is_oth), 
  ncol(d))


#' ## Factor Variables
#' I will inspect each factor variable, looking for
#' near-zero-variance predictors.  I won't
#' remove them here, but I might remove them in the modeling pre-process step if
#' they cause zero-variance in cv folds (see discussion 
#' [here](https://www.r-bloggers.com/near-zero-variance-predictors-should-we-remove-them/)).
#' `caret` has a nice function `nearZeroVar()` to identify near-zero variance predictors.  It defines near-zero variance
#' as a frequency ratio (ratio of the most common value frequency to the second most common 
#' value frequency) >= 19 and a percent of unique values <= 10% (see [caret package 
#' documentation](https://topepo.github.io/caret/index.html)).
#' (DataCamp course [Machine Learning Toolbox](https://campus.datacamp.com/courses/machine-learning-toolbox/preprocessing-your-data?ex=13)
#' suggests more aggressive thresholds of frequency ratio >= 2 and percent
#' of unique values <= 20%).  
#' 

(nzv <- nearZeroVar(d[, col_is_ord | col_is_nom], 
                    saveMetrics= TRUE,
                    freqCut = 99/5, uniqueCut = 10))
nzv_colnames <- rownames(nzv[nzv$nzv == TRUE,])

#' There are `r nrow(nzv[nzv$nzv == TRUE,])` factor variables with near-zero variance.  
#' Here are distribution bar plots of the ordinal factor variables.  The five NZVs are 
#' colored gold.

#+ fig.height = 16
p <- map(colnames(d[,col_is_ord]),
         ~ ggplot(d, aes_string(x = .x)) +
           geom_bar(fill = ifelse(.x %in% nzv_colnames, "goldenrod", "cadetblue")) +
           labs(y = "", x = "", title = .x) +
           theme(axis.text.y=element_blank(),
                 plot.title = element_text(size = 10)))
exec(grid.arrange, ncol = 4, !!!p)

#' And here are the distribution bar plots of the nominal factor variables.  The ten 
#' NZVs are colored gold.

#+ fig.height = 16
p <- map(colnames(d[,col_is_nom]),
         ~ ggplot(d, aes_string(x = .x)) +
           geom_bar(fill = ifelse(.x %in% nzv_colnames, "goldenrod", "cadetblue")) +
           labs(y = "", x = "", title = .x) +
           theme(axis.text.y=element_blank(),
                 plot.title = element_text(size = 10)))
exec(grid.arrange, ncol = 4, !!!p)

#' ## Quantitative Variables
#' Skew can contribute to violation of linearity in linear regressions.  I'll 
#' check which variables have significant skew.  Skew between 0.5 and 1.0 
#' is generally considered moderate, and skew greater than 1 severe.  In the 
#' following charts, the moderately skewed predictors are colored gold and the
#' severely skewed predictors are colored red.

#+ message=FALSE, fig.height = 16
col_skew <- map(d[, col_is_num], skewness) %>% unlist()
col_skew_is_mod <- names(col_skew[abs(col_skew) > .5 & abs(col_skew) <= 1.0])
col_skew_is_high <- names(col_skew[abs(col_skew) > 1.0])
p <- map(colnames(d[,col_is_num]),
         ~ ggplot(d, aes_string(x = .x)) +
           geom_histogram(fill = case_when(.x %in% col_skew_is_mod ~ "goldenrod", 
                                           .x %in% col_skew_is_high ~ "orangered4",
                                           TRUE ~ "cadetblue")) +
           labs(y = "", x = "", title = .x) +
           theme(axis.text.y=element_blank(),
                 plot.title = element_text(size = 10)))
exec(grid.arrange, ncol = 4, !!!p)

#' Almost all the distributions seem reasonable.  Only six of the 34 numeric variables 
#' are not skewed.  I'll want to transform these to help create linear relationships
#' with the response variable.  `YearRemodAdd` seems odd though - why the spike in
#' remodels at the earliest year?

ggplot(d, aes(x = YearBuilt, y = YearRemodAdd)) +
  geom_jitter(alpha = 0.1, color = "cadetblue") +
  geom_abline(na.rm = TRUE) +
  coord_cartesian(xlim = c(1895, 2015), ylim = c(1895, 2015))

#' For some reason, `YearRemodAdd` is floored at 1950.  I'm going to change all 
#' instances of `YearRemodAdd` = 1950 to the value of `YearBuilt`. 
d$YearRemodAdd = ifelse(d$YearRemodAdd > 1950, d$YearRemodAdd, d$YearBuilt)

ggplot(d, aes(x = YearBuilt, y = YearRemodAdd)) +
  geom_jitter(alpha = 0.1, color = "cadetblue") +
  geom_abline(na.rm = TRUE) +
  coord_cartesian(xlim = c(1895, 2015), ylim = c(1895, 2015))

#' # Bi-Variate Analysis
#' 
#' In this section I will look at inter-variable relationships. For factor variables, I am 
#' interested in which levels have significantly different mean `SalePrice` values. For 
#' quantitative variables, I am looking for linear relationships with `SalePrice` and 
#' low correlations with each other.
#' 
#' ## Factor Variables 
#' 
#' For each factor variable I will construct box-plot of `SalePrice` and conduct 
#' a post-hoc test on the factor levels to determine which groups differ from 
#' others.  For groups with no significant difference, I can collapse the levels,
#' or leave them alone and allow the penalization models to reduce their coefficients
#' or the tree models to disregard them.
#' 
#' Here are the nominal factor variables.
#' 
for(i in names(col_is_nom[col_is_nom == TRUE | col_is_ord])) {
  myHSDLvl <- aov(as.formula(paste0("SalePrice ~ ", i)), data = d[d$Set == "train",]) %>%
    HSD.test(trt = i, console = FALSE) %>%
    .$groups %>%
    as.data.frame() %>%
    rownames_to_column()
  colnames(myHSDLvl) <- c(i, "SalePrice", "groups")
  myHSDLvl$SalePrice <- 1000
  
  p <- d %>% 
    filter(Set == "train") %>%
    ggplot(aes_string(y = "SalePrice", x = i)) + 
    geom_jitter(aes_string(color = i), width = 0.2) + 
    geom_boxplot(aes_string(fill = i), alpha = 0.5, outlier.shape = NA) + 
    stat_boxplot(geom = "errorbar", width = 0.4) +
    geom_label(data = myHSDLvl, aes_string(label = "groups")) +
    scale_y_continuous(labels = scales::comma) +
    theme_minimal() +
    theme(legend.position = "none") +
    labs(title = paste("Sale Price vs", i), x = "", y = "") +
    theme(plot.title = element_text(size = 11))
  print(p)
}

#' ## Quantitative Variables
#' 
#' For each quantitative variable, I am interested in its correlation with `SalePrice` and
#' with other quantitative variables.
#' 
#+ fig.width = 8
corrplot(cor(d[, col_is_num]),
         title = "Quantitative Variable Correlation Matrix")

#' # Add Features
#' 
#' I will apply my (scant) subject matter knowledge to propose new variables derived from others.  
#' 
#' ## Collapse NZV
#' I'll propose variables that collapse levels of the NZV factor vars identified above (at least
#' for the ones that seem to make sense to collapse).
#' 
nzv_colnames
d <- d %>%
  mutate(HasAlley = as.factor(ifelse(as.character(Alley) == "NA", 0, 1)),
         IsFunctional = as.factor(ifelse(as.character(Functional) == "Typ", 1, 0)),
         GasHeating = as.factor(ifelse(as.character(Heating) == "GasA", 1, 0)),
         LvlLand = as.factor(ifelse(as.character(LandContour) == "Lvl", 1, 0)),
         GtlSlope = as.factor(ifelse(as.character(LandSlope) == "Gtl", 1, 0)),
         HasMiscFtr = as.factor(ifelse(as.character(MiscFeature) == "NA", 0, 1))) %>% data.frame()

#' ## Summary Features
#' I'll combine features summary features.
#' 
d <- d %>%
  mutate(TotalSF = TotalBsmtSF + GrLivArea,
         TotalBath = BsmtFullBath + 0.5 * BsmtHalfBath + FullBath + 0.5 * HalfBath,
         AbvGrdSFperRm = GrLivArea / TotRmsAbvGrd,
         HasBsmt = as.factor(ifelse(TotalBsmtSF == 0, 0, 1)),
         HasGarage = as.factor(ifelse(GarageArea == 0, 0, 1))
  ) %>% data.frame()

#' ## Log(SalePrice)
#' I'll log-transform `SalePrice` because that is what we are predicting, and because
#' `SalePrice` is skewed.
#' 
d <- d %>%
  mutate(logSalePrice = log(SalePrice)) %>% 
  data.frame()

#' Now that the data tranformation tasks are complete, I'll split `d` back into `d.train` and `d.test`.
#' 
d <- d %>% select(Set, Id, SalePrice, logSalePrice, sort(names(d))) %>% data.frame()
d.train <- d[d$Set == "train", which(colnames(d) != "Set")]
d.test <- d[d$Set == "test", which(colnames(d) != "Set")]


#' # Handle Outliers

#' Now I'll look at influential outliers.  When building a predictive model, I only need to address 
#' influential outliers for variables that are likely to be important in the final model.  I'll identify 
#' the most likely important variables with a correlation matrix of the numerical predictors.  For every 
#' variable with correlation >= 0.5 (a moderate correlation), I'll check the variable distribution to 
#' see whether any influential outliers exist.
#' 
#+ fig.height = 3, warning=FALSE, message=FALSE
col_is_num <- map(d.train, ~ class(.x)[1] %in% c("integer", "numeric")) %>% unlist() 
col_is_num <- col_is_num == TRUE & !(names(col_is_num) %in% c("Id", "logSalePrice"))
(var_num.cor <- cor(d.train[, col_is_num]) %>%
    data.frame() %>%
    select(SalePrice) %>%
    data.frame(Variable = rownames(.), Corr = .) %>%
    filter(abs(SalePrice) >= 0.5 & Variable != "SalePrice") %>%
    arrange(desc(SalePrice)))

for (x in as.character(var_num.cor$Variable)) {
  p1 <- ggplot(d.train, aes_string(x = x)) + geom_histogram() + labs(x = x)
  p2 <- ggplot(d.train, aes_string(x = x, y = "SalePrice")) + geom_point() + labs(x = x) +
    geom_smooth(method = "lm", se = TRUE, na.rm = TRUE)
  grid.arrange(p1, p2, ncol = 2)
}

#' I count six potential influential outliers.  They are based on the union of 
#' the conditions in the filter condition below.  I can see why the author 
#' recommends dropping `GrLivArea` > 4000 - it captures 4 of them.
#' 
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
