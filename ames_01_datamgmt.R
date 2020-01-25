#' ---
#' title: "Kaggle - House Prices"
#' subtitle: "Step 1: Data Management"
#' author: "Gimme the Prize"
#' output: 
#'    html_document:
#'      theme: flatly
#'      toc: true
#'      highlight: haddock
#'      fig_width: 9
#'      fig_caption: false
#' ---

#' This is an analysis of the *Ames* dataset for the Kaggle competition 
#' [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques).  
#' Kaggle's House Prices competition challenges participants to predict the final sale price of 1,459 homes sold in Ames, 
#' IA from 2006 to 2010. The original dataset was published by Dean De Cock in [Ames, Iowa: Alternative to the Boston 
#' Housing Data as an End of Semester Regression Project](https://ww2.amstat.org/publications/jse/v19n3/decock.pdf).  The 
#' source data set for the competition contains 2,919 observations and 79 explanatory variables (23 nominal, 
#' 23 ordinal, 14 discrete, and 19 continuous) involved in assessing home values. Kaggle segmented the data set into a 
#' training set consisting of 1,460 observations and a test data set consisting of 1,459 observations with the final 
#' sale price excluded.  Competitors build a predictive model with the training set, then apply the model to the 
#' test set to produce a submission file consisting of the observation id and the predicted sale price.  Kaggle evaluates 
#' submissions on the root-mean-squared-error (RMSE) between the log of the predicted sales price and the log of the 
#' observed sales price.
#' 
#' This document addresses initial data management: data cleaning and imputation.
#' 
#' 
#' # Setup

#+ echo = FALSE
#+ setup, message = FALSE, warning = FALSE
library(tidyverse)
library(gridExtra)
library(mice)

#' # Load Data

#' Load train and test, then combine to create a single data set for exploration and feature engineering.

#+ load_data, results = 'hide', message = FALSE, warning = FALSE
# "NA" is an actual data label - only allow empty string "" for `NA`
d.train <- read_csv("./train.csv", na = c("")) %>% mutate(Set = "train")
d.train <- rename(d.train, !!c(X1stFlrSF = "1stFlrSF",
                               X2ndFlrSF = "2ndFlrSF",
                               X3SsnPorch = "3SsnPorch"))
d.test <- read_csv("./test.csv", na = c("")) %>% mutate(SalePrice = NA, Set = "test")
d.test <- rename(d.test, !!c(X1stFlrSF = "1stFlrSF",
                             X2ndFlrSF = "2ndFlrSF",
                             X3SsnPorch = "3SsnPorch"))
d <- rbind(d.train, d.test)
d <- d %>% select(Set, Id, SalePrice, sort(names(d))) %>% data.frame()

#' Define the ordinal factor variable levels from the 
#' [codebook](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/download/data_description.txt).
#' I count 19 ordinal variables, not 23 as indicated by De Cock.  
#' `OverallQual` and `OverallCond` could be ordinal or numeric.  With 
#' experimentation, it appears they perform slightly better as numeric, so I
#' removed them from this list.
OF.levels <- list(
  BsmtCond = c('NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'),
  BsmtExposure = c('NA', 'No', 'Mn', 'Av', 'Gd'),
  BsmtFinType1 = c('NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'),
  BsmtFinType2 = c('NA', 'Unf', 'LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ'),
  BsmtQual = c('NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'),
  Electrical = c('FuseP', 'FuseF', 'FuseA', 'Mix', 'SBrkr'),
  ExterQual = c('Po', 'Fa', 'TA', 'Gd', 'Ex'),
  ExterCond = c('Po', 'Fa', 'TA', 'Gd', 'Ex'),
  Fence = c('NA', 'MnWw', 'MnPrv', 'GdWo', 'GdPrv'),
  FireplaceQu = c('NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'),
  Functional = c('Sal', 'Sev', 'Maj2', 'Maj1', 'Mod', 'Min2', 'Min1', 'Typ'),
  GarageCond = c('NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'),
  GarageFinish = c('NA', 'Unf', 'RFn', 'Fin'),
  GarageQual = c('NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex'),
  HeatingQC = c('Po', 'Fa', 'TA', 'Gd', 'Ex'),
  KitchenQual = c('Po', 'Fa', 'TA', 'Gd', 'Ex'),
  PoolQC = c('NA', 'Po', 'Fa', 'TA', 'Gd', 'Ex')
)

#' Define the nominal factor variable levels.  I count 28 nominal variables, not 23 as indicated by De Cock
UF.levels <- list(
  Alley = c('Grvl', 'Pave', 'NA'),
  BldgType = c('1Fam', '2FmCon', 'Duplx', 'TwnhsE', 'TwnhsI'),
  CentralAir = c('N', 'Y'),
  Condition1 = c('Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'),
  Condition2 = c('Artery', 'Feedr', 'Norm', 'RRNn', 'RRAn', 'PosN', 'PosA', 'RRNe', 'RRAe'),
  Exterior1st = c('AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard',
                  'ImStucc','MetalSd','Other','Plywood','PreCast','Stone','Stucco',
                  'VinylSd','Wd Sdng','WdShing'),
  Exterior2nd = c('AsbShng','AsphShn','BrkComm','BrkFace','CBlock','CemntBd','HdBoard',
                  'ImStucc','MetalSd','Other','Plywood','PreCast','Stone','Stucco',
                  'VinylSd','Wd Sdng','WdShing'),
  Foundation = c('BrkTil', 'CBlock', 'PConc', 'Slab', 'Stone', 'Wood'),
  GarageType = c('2Types', 'Attchd', 'Basment', 'BuiltIn', 'CarPort', 'Detchd', 'NA'),
  Heating = c('Floor', 'GasA', 'GasW', 'Grav', 'OthW', 'Wall'),
  HouseStyle = c('1Story', '1.5Fin', '1.5Unf', '2Story', '2.5Fin', '2.5Unf', 'SFoyer', 'SLvl'),
  LandSlope = c('Gtl', 'Mod', 'Sev'),
  LotConfig = c('Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'),
  LandContour = c('Lvl', 'Bnk', 'HLS', 'Low'),
  LotShape = c('Reg', 'IR1', 'IR2', 'IR3'),
  MasVnrType = c('BrkCmn', 'BrkFace', 'CBlock', 'Stone', 'None'),
  MoSold = 1:12,
  MiscFeature = c('Elev', 'Gar2', 'Othr', 'Shed', 'TenC', 'NA'),
  MSSubClass = c(20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 150, 160, 180, 190),
  MSZoning = c('A', 'C', 'FV', 'I', 'RH', 'RL', 'RP', 'RM'),
  Neighborhood = c('Blmngtn','Blueste','BrDale','BrkSide','ClearCr','CollgCr',
                   'Crawfor','Edwards','Gilbert','IDOTRR','MeadowV','Mitchel',
                   'Names','NoRidge','NPkVill','NridgHt','NWAmes','OldTown',
                   'SWISU','Sawyer','SawyerW','Somerst','StoneBr','Timber',
                   'Veenker'),
  PavedDrive = c('Y', 'P', 'N'),
  RoofMatl = c('ClyTile','CompShg','Membran','Metal','Roll','Tar&Grv','WdShake','WdShngl'),
  RoofStyle = c('Flat', 'Gable', 'Gambrel', 'Hip', 'Mansard', 'Shed'),
  SaleCondition = c('Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'),
  SaleType = c('WD','CWD','VWD','New','COD','Con','ConLw','ConLI','ConLD','Oth'),
  Street = c('Grvl', 'Pave'),
  Utilities = c('AllPub', 'NoSwr', 'NoSeWa', 'ELO')
)

#' # Clean Data
#' 
#' I'll look at each feature and adjust the data type and/or factor levels as necessary so that the data is reconciled with the codebook.
#' 

d$Id <- as.numeric(d$Id)
d$SalePrice <- as.numeric(d$SalePrice)
d$Alley <- parse_factor(d$Alley, levels = UF.levels$Alley)
d$BedroomAbvGr <- as.numeric(d$BedroomAbvGr)

#' BldgType levels do not match values in data description file.
#' Doesn't look like typos - the data codebook just does not match.
#' Use the data file levels instead of the codebook levels.
setdiff(unique(d$BldgType), OF.levels$BldgType)
UF.levels$BldgType <- unique(d$BldgType)
d$BldgType <- parse_factor(d$BldgType, levels = UF.levels$BldgType)

#'
#+ warning=FALSE
d$BsmtCond <- parse_factor(d$BsmtCond, levels = OF.levels$BsmtCond, ordered = TRUE) %>% fct_explicit_na("NA")
d$BsmtExposure <- parse_factor(d$BsmtExposure, levels = OF.levels$BsmtExposure, ordered = TRUE)
d$BsmtFinSF1 <- as.numeric(d$BsmtFinSF1)
d$BsmtFinSF2 <- as.numeric(d$BsmtFinSF2)
d$BsmtFinType1 <- parse_factor(d$BsmtFinType1, levels = OF.levels$BsmtFinType1, ordered = TRUE)
d$BsmtFinType2 <- parse_factor(d$BsmtFinType2, levels = OF.levels$BsmtFinType2, ordered = TRUE)
d$BsmtFullBath <- as.numeric(d$BsmtFullBath)
d$BsmtHalfBath <- as.numeric(d$BsmtHalfBath)
d$BsmtQual <- parse_factor(d$BsmtQual, levels = OF.levels$BsmtQual, ordered = TRUE) %>% fct_explicit_na("NA")
d$BsmtUnfSF <- as.numeric(d$BsmtUnfSF)
d$CentralAir <- parse_factor(d$CentralAir, levels = UF.levels$CentralAir)
d$Condition1 <- parse_factor(d$Condition1, levels = UF.levels$Condition1)
d$Condition2 <- parse_factor(d$Condition2, levels = UF.levels$Condition2)

#' `Electrical` includes `NA`s coded as "NA". 
d$Electrical <- ifelse(d$Electrical == "NA", NA, d$Electrical)
d$Electrical <- parse_factor(d$Electrical, levels = OF.levels$Electrical, ordered = T, include_na = F)

#'
d$EnclosedPorch <- as.numeric(d$EnclosedPorch)
d$ExterCond <- parse_factor(d$ExterCond, levels = OF.levels$ExterCond, ordered = T)

#' `Exterior1st` includes `NA`s coded as "NA". 
d$Exterior1st <- ifelse(d$Exterior1st == "NA", NA, d$Exterior1st)
d$Exterior1st <- parse_factor(d$Exterior1st, levels = UF.levels$Exterior1st, include_na = F)

#' `Exterior2nd` includes `NA`s coded as "NA". 
#' Some values in `Exterior2nd` do not match the levels in the codebook.
d$Exterior2nd <- ifelse(d$Exterior2nd == "NA", NA, d$Exterior2nd)
d <- d %>% mutate(Exterior2nd = case_when(Exterior2nd == "Wd Shng" ~ "WdShing",
                                          Exterior2nd == "CmentBd" ~ "CemntBd",
                                          Exterior2nd == "Brk Cmn" ~ "BrkFace",
                                          TRUE ~ Exterior2nd))
d$Exterior2nd <- parse_factor(d$Exterior2nd, levels = UF.levels$Exterior2nd, include_na = F)

#'
d$ExterQual <- parse_factor(d$ExterQual, levels = OF.levels$ExterQual, ordered = T)
d$Fence <- parse_factor(d$Fence, levels = OF.levels$Fence, ordered = T) %>% fct_explicit_na("NF")
d$FireplaceQu <- parse_factor(d$FireplaceQu, levels = OF.levels$FireplaceQu, ordered = T)
d$Fireplaces <- as.numeric(d$Fireplaces)
d$Foundation <- parse_factor(d$Foundation, levels = UF.levels$Foundation)
d$FullBath <- as.numeric(d$FullBath)

#' `Functional` includes `NA`s coded as "NA". 
d$Functional <- ifelse(d$Functional == "NA", NA, d$Functional)
d$Functional <- parse_factor(d$Functional, levels = OF.levels$Functional, ordered = T, include_na = F)

#'
#+ warning=FALSE
d$GarageArea <- as.numeric(d$GarageArea)
d$GarageCars <- as.numeric(d$GarageCars)
d$GarageCond <- parse_factor(d$GarageCond, levels = OF.levels$GarageCond, ordered = T)
d$GarageFinish <- parse_factor(d$GarageFinish, levels = OF.levels$GarageFinish, ordered = T)
d$GarageQual <- parse_factor(d$GarageQual, levels = OF.levels$GarageQual, ordered = T)
d$GarageType <- parse_factor(d$GarageType, levels = UF.levels$GarageType)
d$GarageYrBlt <- as.numeric(d$GarageYrBlt)
d$GrLivArea <- as.numeric(d$GrLivArea)
d$HalfBath <- as.numeric(d$HalfBath)
d$Heating <- parse_factor(d$Heating, levels = UF.levels$Heating)
d$HeatingQC <- parse_factor(d$HeatingQC, levels = OF.levels$HeatingQC, ordered = T)
d$HouseStyle <- parse_factor(d$HouseStyle, levels = UF.levels$HouseStyle)
d$KitchenAbvGr <- as.numeric(d$KitchenAbvGr)

#' `KitchenQual` includes `NA`s coded as "NA". 
d$KitchenQual <- ifelse(d$KitchenQual == "NA", NA, d$KitchenQual)
d$KitchenQual <- parse_factor(d$KitchenQual, levels = OF.levels$KitchenQual, ordered = T, include_na = F)

#'
#+ warning=FALSE
d$LandContour <- parse_factor(d$LandContour, levels = UF.levels$LandContour)
d$LandSlope <- parse_factor(d$LandSlope, levels = UF.levels$LandSlope)
d$LotArea <- as.numeric(d$LotArea)
d$LotConfig <- parse_factor(d$LotConfig, levels = UF.levels$LotConfig)
d$LotFrontage <- as.numeric(d$LotFrontage)
d$LotShape <- parse_factor(d$LotShape, levels = UF.levels$LotShape)
d$LowQualFinSF <- as.numeric(d$LowQualFinSF)
d$MasVnrArea <- as.numeric(d$MasVnrArea)

#' `MasVnrType` includes `NA`s coded as "NA". 
d$MasVnrType <- ifelse(d$MasVnrType == "NA", NA, d$MasVnrType)
d$MasVnrType <- parse_factor(d$MasVnrType, levels = UF.levels$MasVnrType, include_na = F)

#'
d$MiscFeature <- parse_factor(d$MiscFeature, levels = UF.levels$MiscFeature) %>% fct_explicit_na("None")
d$MiscVal <- as.numeric(d$MiscVal)
d$MoSold <- parse_factor(as.character(d$MoSold), levels = UF.levels$MoSold)
d$MSSubClass <- parse_factor(as.character(d$MSSubClass), levels = UF.levels$MSSubClass)

#' `MSZoning` includes `NA`s coded as "NA". 
#' `MSZoning` factor level "C" is coded as "C (all)" in the data.  Correct the data.
d$MSZoning <- ifelse(d$MSZoning == "NA", NA, d$MSZoning)
d$MSZoning <- parse_factor(ifelse(d$MSZoning == "C (all)", "C", d$MSZoning), levels = UF.levels$MSZoning, include_na = F)

#' `Neighborhood` factor level "NAmes" is coded as "Names" in the data.  Correct the codebook.
UF.levels$Neighborhood[UF.levels$Neighborhood == "Names"] <- "NAmes"
d$Neighborhood <- parse_factor(d$Neighborhood, levels = UF.levels$Neighborhood)

#'
d$OpenPorchSF <- as.numeric(d$OpenPorchSF)
#d$OverallCond <- parse_factor(as.character(d$OverallCond), levels = OF.levels$OverallCond, ordered = T)
#d$OverallQual <- parse_factor(as.character(d$OverallQual), levels = OF.levels$OverallQual, ordered = T)
d$PavedDrive <- parse_factor(d$PavedDrive, levels = UF.levels$PavedDrive)
d$PoolArea <- as.numeric(d$PoolArea)
d$PoolQC <- parse_factor(d$PoolQC, levels = OF.levels$PoolQC, ordered = T) %>% fct_explicit_na("NP")
d$RoofMatl <- parse_factor(d$RoofMatl, levels = UF.levels$RoofMatl)
d$RoofStyle <- parse_factor(d$RoofStyle, levels = UF.levels$RoofStyle)
d$SaleCondition <- parse_factor(d$SaleCondition, levels = UF.levels$SaleCondition)

#' `SaleType` includes `NA`s coded as "NA". 
d$SaleType <- ifelse(d$SaleType == "NA", NA, d$SaleType)
d$SaleType <- parse_factor(d$SaleType, levels = UF.levels$SaleType, include_na = F)

#'
#+ warning=FALSE
d$ScreenPorch <- as.numeric(d$ScreenPorch)
d$Street <- parse_factor(d$Street, levels = UF.levels$Street)
d$TotalBsmtSF <- as.numeric(d$TotalBsmtSF)
d$TotRmsAbvGrd <- as.numeric(d$TotRmsAbvGrd)

#' `Utilities` includes `NA`s coded as "NA". 
d$Utilities <- ifelse(d$Utilities == "NA", NA, d$Utilities)
d$Utilities <- parse_factor(d$Utilities, levels = UF.levels$Utilities, include_na = F)

#'
d$WoodDeckSF <- as.numeric(d$WoodDeckSF)
d$X1stFlrSF <- as.numeric(d$X1stFlrSF)
d$X2ndFlrSF <- as.numeric(d$X2ndFlrSF)
d$X3SsnPorch <- as.numeric(d$X3SsnPorch)
d$YearBuilt <- as.numeric(d$YearBuilt)
d$YearRemodAdd <- as.numeric(d$YearRemodAdd)
d$YrSold <- as.numeric(d$YrSold)

#' How does it look?
skimr::skim_to_wide(d)[, c(1:3, 9, 14, 16, 18, 19)] %>% knitr::kable()

#' Factors and numerics are accounted for correctly.  But there are some data issues to address
#' in the next section.
#'
#' # Manage Data

#' Now that the data is reconciled with the codebook, I will address data structure issues, correct obvious
#' errors, and impute values for `NA`. Browsing through the codebook and looking at the value ranges, I see 
#' a few issues.
#' 
#' ## `Condition1` and `Condition2`
#' 
#' Similarly, variables `Condition1` and `Condition2` are really two variables checking for the existance of a
#' major road (Artery or feeder), railroad (close to or adjacent to east-west or north-south railroad), or 
#' some positive feature like green space.  I'll create vars for the presence of each one (0 = not present
#' in either `Condition1` or `Condition2`, 1 = present in one or the other, and 2 = present in both).)
d$CondRoad <- factor(ifelse(d$Condition1 %in% c('Artery', 'Feedr'), 1, 0) +
                       ifelse(d$Condition2 %in% c('Artery', 'Feedr'), 1, 0),
                     ordered = TRUE)
d$CondRail <- factor(ifelse(d$Condition1 %in% c('RRNn', 'RRAn', 'RRNe', 'RRAe'), 1, 0) +
                       ifelse(d$Condition2 %in% c('RRNn', 'RRAn', 'RRNe', 'RRAe'), 1, 0),
                     ordered = TRUE)
d$CondPos <- factor(ifelse(d$Condition1 %in% c('PosN', 'PosA'), 1, 0) +
                      ifelse(d$Condition2 %in% c('PosN', 'PosA'), 1, 0),
                    ordered = TRUE)
d$Cond <- factor(case_when(d$Condition1 %in% c('RRNn', 'RRAn', 'RRNe', 'RRAe', 'Artery', 'Feedr') ~ -1,
                           d$Condition2 %in% c('RRNn', 'RRAn', 'RRNe', 'RRAe', 'Artery', 'Feedr') ~ -1,
                           d$Condition1 %in% c('PosN', 'PosA') ~ 1,
                           d$Condition2 %in% c('PosN', 'PosA') ~ 1,
                           TRUE ~ 0),
                 ordered = TRUE)

#+ warning = FALSE
p1 <- d %>% ggplot(aes(x = CondRoad, y = SalePrice)) + 
  geom_jitter(aes(color = CondRoad), width = 0.2) +
  geom_boxplot(aes(fill = CondRoad), alpha = 0.5, outlier.shape = NA) + 
  stat_boxplot(geom = "errorbar", width = 0.4) + 
  theme(legend.position = "none")
p2 <- d %>% ggplot(aes(x = CondRail, y = SalePrice)) + 
  geom_jitter(aes(color = CondRail), width = 0.2) +
  geom_boxplot(aes(fill = CondRail), alpha = 0.5, outlier.shape = NA) + 
  stat_boxplot(geom = "errorbar", width = 0.4) + 
  theme(legend.position = "none")
p3 <- d %>% ggplot(aes(x = CondPos, y = SalePrice)) + 
  geom_jitter(aes(color = CondPos), width = 0.2) +
  geom_boxplot(aes(fill = CondPos), alpha = 0.5, outlier.shape = NA) + 
  stat_boxplot(geom = "errorbar", width = 0.4) + 
  theme(legend.position = "none")
grid.arrange(p1, p2, p3, nrow = 1)
ggplot(d, aes(x = Cond, y = SalePrice)) + 
  geom_jitter(aes(color = Cond), width = 0.2) +
  geom_boxplot(aes(fill = Cond), alpha = 0.5, outlier.shape = NA) + 
  stat_boxplot(geom = "errorbar", width = 0.4) + 
  theme(legend.position = "none")

#' ## `GarageYrBlt`
#' 
#' `GarageYrBlt` has maximum value 2207.  
d %>% filter(GarageYrBlt > 2010) %>% select(GarageYrBlt, YearBuilt)

#' I'll bet a nickle the garage from this home was built in 2007, not 2207.
d$GarageYrBlt <- ifelse(d$GarageYrBlt == 2207, 2007, d$GarageYrBlt)

#' ## Impute `NA`
#'
#' There are still several variables with missing values.

d %>% 
  summarize_all(~ sum(is.na(.))/length(.)) %>% 
  gather(key = "Column", value = "% NA") %>% 
  filter(`% NA` > 0 & Column != 'SalePrice') %>%
  mutate(ColType = map_chr(Column, ~ paste(class(d[[.]]), collapse = " "))) %>%
  select(Column, ColType, `% NA`) %>%
  arrange(desc(`% NA`))

#' Are there any patterns in the missing data?  This plot shows the row count in the left col, which vars have missing values
#' in the colors (red = missing), and number of vars with missing values in the right axis.  E.g., 11 observations are have 
#' `NA` for both `LotFrontage` and `GarageYrBlt`.

#+ fig.height = 12
invisible(md.pattern(d[,which(colnames(d) != "SalePrice")] %>% 
                       select_if(~ any(is.na(.))), 
                     rotate.names = TRUE))

#' In one instance, `BsmtFinSF1`, `BsmtFinSF2`, `BsmtUnfSF`, `TotalBsmtSF`, `BsmtFullBath`, 
#' and `BsmtHalfBath` are `NA` together.  They are probably `NA` because the home has no basement.
#' I'll map these `NA`s to 0.
idx <- which(is.na(d$BsmtFullBath) & is.na(d$BsmtHalfBath) & is.na(d$BsmtFinSF1) &
               is.na(d$BsmtFinSF2) & is.na(d$BsmtUnfSF) & is.na(d$BsmtUnfSF) & is.na(d$TotalBsmtSF))
d[idx, "BsmtFullBath"] <- 0
d[idx, "BsmtHalfBath"] <- 0
d[idx, "BsmtFinSF1"] <- 0
d[idx, "BsmtFinSF2"] <- 0
d[idx, "BsmtUnfSF"] <- 0
d[idx, "TotalBsmtSF"] <- 0
rm(idx)

#' When `MasVnrArea` is `NA`, so is `MasVnrType`.  There is one instance where the opposite is not true (`MasVnrType` 
#' is `NA`, but `MasVnrArea` is 198.  For instances where `MasVnrArea` is `NA`, I'll set `MasVnrArea` to 0 and  
#' `MasVnrType` to `None`.  

# d %>% filter(is.na(MasVnrType)) %>% select(MasVnrArea)
d <- d %>% 
  mutate(MasVnrType = ifelse(is.na(MasVnrArea) & is.na(MasVnrType), "None", as.character(MasVnrType)),
         MasVnrArea = ifelse(is.na(MasVnrArea) & is.na(MasVnrType), 0, MasVnrArea))
d$MasVnrType <- parse_factor(d$MasVnrType, levels = UF.levels$MasVnrType)

#' In one istance, `GarageArea`, `GarageCars`, and `GarageYrBlt` are `NA` together.  
#' They are probably `NA` because the home has no garage.
idx <- which(is.na(d$GarageArea) & is.na(d$GarageCars) & is.na(d$GarageYrBlt))
d[idx, "GarageArea"] <- 0
d[idx, "GarageCars"] <- 0
d[idx, "GarageYrBlt"] <- min(d$GarageYrBlt, na.rm = TRUE)
rm(idx)

#' `GarageYrBlt` is `NA` in 159 observations.  Is this situation similar to `NA` basement vars?
skimr::skim_to_wide(d[is.na(d$GarageYrBlt), 
                      which(colnames(d) %in% c("GarageArea", "GarageCars", "GarageCond", 
                                               "GarageFinish", "GarageQual", "GarageType"))])
#' Seems like it.  In cases where all seven garage vars are "NA" or `NA`, set `GarageYrBlt` to the 
#' oldest year in the data set under the logic that no garage is about as valuable as the oldest 
#' possible garage.
d$GarageYrBlt <- case_when(is.na(d$GarageYrBlt) & d$GarageArea == 0 & d$GarageCars == 0 &
                             d$GarageCond == "NA" & d$GarageFinish == "NA" & d$GarageQual == "NA" &
                             d$GarageType == "NA" ~ min(d$GarageYrBlt, na.rm = TRUE),
                           TRUE ~ d$GarageYrBlt)

#' Check the pattern one more time. 
#+ fig.height = 12
invisible(md.pattern(d[,which(colnames(d) != "SalePrice")] %>% 
                       select_if(~ any(is.na(.))), 
                     rotate.names = TRUE))

#' Use MICE to impute values for `NA`.  (see discussion of algorithm 
#' [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3074241/). 

imp <- mice(d, 
            predictorMatrix = quickpred(d, exclude = c("Set", "Id")), 
            seed = 1970, 
            method = "cart", 
            maxit = 1)

densityplot(imp)
xyplot(imp, LotFrontage ~ LotArea, col=mdc(1:2), pch=20, cex=1.5)
xyplot(imp, MasVnrArea ~ GrLivArea, col=mdc(1:2), pch=20, cex=1.5)

#' Complete the data set with missing values replaced by imputations.
d <- complete(imp)

#' Am I done?  Yes, only `SalePrice` from the test dataset remains.
#' 
d %>% skimr::skim_to_wide() %>% filter(as.numeric(missing) > 0)

#' ...but set the test set `SalePrice` values back to `NA`!
d$SalePrice <- ifelse(d$Set == "test", NA, d$SalePrice)

#' Here is one more look at the data set.
#' 
skimr::skim_to_wide(d)[, c(1:3, 9, 14, 16, 18, 19)] %>% knitr::kable()

#' # Save Work
#' 
#' Save the data as an input to the next step, exploratory data analysis

saveRDS(d, file = "./ames_01.RDS")
