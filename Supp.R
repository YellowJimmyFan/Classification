---
title: "Supp code"
output: pdf_document
---

# Preliminary
```{r message=FALSE, warning=FALSE}
rm(list=ls())
set.seed("441")
library(nnet)
library(e1071)
library(glmnet)
library(xgboost)
library(randomForestSRC)
# a function to convert all columns to numeric
convert_to_numeric <- function(df) {
  df[] <- lapply(df, function(col) {
    if (is.factor(col) || is.character(col)) {
      as.numeric(as.factor(col))
    } else {
      as.numeric(col)
    }
  })
  return(df)
}
```

# Load the data and remove redundant columns
```{r}
edu_train <- read.csv("module_Education_train_set.csv")
edu_test <- read.csv("module_Education_test_set.csv")
edu_train$psu_hh_idcode <- with(edu_train, paste(psu, hh, idcode, sep = "_"))
edu_train$psu <- NULL
edu_train$hh <- NULL
edu_train$idcode <- NULL
edu_test$psu_hh_idcode <- with(edu_test, paste(psu, hh, idcode, sep = "_"))
edu_test$psu <- NULL
edu_test$hh <- NULL
edu_test$idcode <- NULL
colnames(edu_train)[1:66] <- paste0("edu", 1:66)
colnames(edu_test)[1:66] <- paste0("edu", 1:66)

Hh_train <- read.csv("module_HouseholdInfo_train_set.csv")
Hh_test <- read.csv("module_HouseholdInfo_test_set.csv")
Hh_train$psu_hh_idcode <- with(Hh_train, paste(psu, hh, idcode, sep = "_"))
Hh_train$psu <- NULL
Hh_train$hh <- NULL
Hh_train$idcode <- NULL
Hh_test$psu_hh_idcode <- with(Hh_test, paste(psu, hh, idcode, sep = "_"))
Hh_test$psu <- NULL
Hh_test$hh <- NULL
Hh_test$idcode <- NULL
colnames(Hh_train)[2:23] <- paste0("hh", 1:22)
colnames(Hh_test)[2:23] <- paste0("hh", 1:22)

SP_train <- read.csv("module_SubjectivePoverty_train_set.csv")
training <- merge(edu_train, Hh_train, by = "psu_hh_idcode")
```

# Merge the data train/test
```{r}
merged_data_train <- merge(training, SP_train, by = "psu_hh_idcode")
merged_data_test <- merge(edu_test, Hh_test, by = "psu_hh_idcode")

## check missing values
missing_columns_train <- colnames(merged_data_train)[colSums(is.na(merged_data_train)) > 0]
missing_columns_train
```

# Treat categorical columns as factors
```{r}
edu_factor_col <- c(paste0("edu", 1:6), paste0("edu", 8:17), 
                    paste0("edu", 19:23), "edu25", "edu28", "edu30", "edu32",
                    paste0("edu", 42:46), "edu48", "edu50", 
                    paste0("edu", 52:54), "edu57", "edu59", "edu61", "edu63",
                    "edu64", "edu66")
hh_factor_col <- c("hh1", "hh2", paste0("hh", 6:8), 
                   paste0("hh", 10:14), paste0("hh", 17:20))
columns_to_factor <- c(edu_factor_col, hh_factor_col)
merged_data_train[columns_to_factor] <- lapply(merged_data_train[columns_to_factor], factor)
merged_data_test[columns_to_factor] <- lapply(merged_data_test[columns_to_factor], factor)
## Add "SKIPPED" factor to all categorical columns training
for (col in c(paste0("edu", 4:66))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[[col]] <- factor(merged_data_train[[col]], 
                                       levels = c(levels(merged_data_train[[col]]), "SKIPPED"))
  }
}

for (col in c(paste0("hh", 4:22))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[[col]] <- factor(merged_data_train[[col]], 
                                       levels = c(levels(merged_data_train[[col]]), "SKIPPED"))
  }
}

## Add "SKIPPED" factor to all categorical columns testing
for (col in c(paste0("edu", 4:66))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[[col]] <- factor(merged_data_test[[col]], 
                                       levels = c(levels(merged_data_test[[col]]), "SKIPPED"))
  }
}

for (col in c(paste0("hh", 4:22))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[[col]] <- factor(merged_data_test[[col]], 
                                       levels = c(levels(merged_data_test[[col]]), "SKIPPED"))
  }
}

```

# More preprocessing categorical
```{r}
############################# training data ####################################
## Handle SKIPPED value for edu64
subset_edu64 <- merged_data_train$edu64 == 2
for (col in c(paste0("edu", 65))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu64, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu61
subset_edu61 <- merged_data_train$edu61 == 2 & !is.na(merged_data_train$edu61)
for (col in c(paste0("edu", 62:63))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu61, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu59
subset_edu59 <- merged_data_train$edu59 == 2
for (col in c(paste0("edu", 60))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu59, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu57
subset_edu57 <- merged_data_train$edu57 == 2
for (col in c(paste0("edu", 58))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu57, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu54
subset_edu54 <- merged_data_train$edu54 == 2
for (col in c(paste0("edu", 55:56))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu54, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu50
subset_edu50 <- merged_data_train$edu50 == 2 & !is.na(merged_data_train$edu50)
for (col in c(paste0("edu", 51:56))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu50, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu48
subset_edu48 <- merged_data_train$edu48 == 4
for (col in c(paste0("edu", 49))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu48, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu46
subset_edu46 <- merged_data_train$edu46 == 2 & !is.na(merged_data_train$edu46)
for (col in c(paste0("edu", 47:49))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu46, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu43
subset_edu43 <- merged_data_train$edu43 == 1 & !is.na(merged_data_train$edu43)
for (col in c(paste0("edu", 44))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu43, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu30
subset_edu30 <- merged_data_train$edu30 == 2
for (col in c(paste0("edu", 31))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu30, col] <- factor("SKIPPED")
  }
}
merged_data_train[is.na(merged_data_train$edu30), "edu30"] <- 2

## Handle SKIPPED value for edu24
subset_edu24_1 <- merged_data_train$edu24 < 5 & !is.na(merged_data_train$edu24)
if (is.factor(merged_data_train[["edu25"]])) {
  merged_data_train[subset_edu24_1, "edu25"] <- factor("SKIPPED")
}

subset_edu24_2 <- merged_data_train$edu24 == 999 & !is.na(merged_data_train$edu24)
for (col in c(paste0("edu", 25:32))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu24_2, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu20
subset_edu20 <- merged_data_train$edu20 %in% 1:2
for (col in c(paste0("edu", 21:66))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu20, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu19
subset_edu19 <- merged_data_train$edu19 == 2 & !is.na(merged_data_train$edu19)
for (col in c(paste0("edu", 20:66))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu19, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu17
subset_edu17 <- merged_data_train$edu17 == 13 & !is.na(merged_data_train$edu17)
for (col in c(paste0("edu", 18:66))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu17, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu16
subset_edu16 <- merged_data_train$edu16 %in% 1:13
for (col in c(paste0("edu", 17:19))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu16, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu15
subset_edu15 <- merged_data_train$edu15 == 1 & !is.na(merged_data_train$edu15)
for (col in c(paste0("edu", 16:20))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu15, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu14
subset_edu14 <- merged_data_train$edu14 == 2 & !is.na(merged_data_train$edu14)
for (col in c(paste0("edu", 15:16))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu14, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu11
subset_edu11 <- merged_data_train$edu11 %in% 1:14
for (col in c(paste0("edu", 12:13))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu11, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu10
subset_edu10 <- merged_data_train$edu10 %in% 1:13
for (col in c(paste0("edu", 11:13))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu10, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu9
subset_edu9 <- merged_data_train$edu9 == 1 & !is.na(merged_data_train$edu9)
for (col in c(paste0("edu", 10:11))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu9, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu8
subset_edu8 <- merged_data_train$edu8 == 2 & !is.na(merged_data_train$edu8)
for (col in c(paste0("edu", 9:10))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu8, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu3
subset_edu3 <- merged_data_train$edu3 == 2
for (col in c(paste0("edu", 4:66))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_edu3, col] <- factor("SKIPPED")
  }
}


## Handle SKIPPED value for hh21
subset_hh21 <- !is.na(merged_data_train$hh21)
if (is.factor(merged_data_train[["hh22"]])) {
  merged_data_train[subset_hh21, "hh22"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh20
subset_hh20 <- merged_data_train$hh20 == 1
if (is.factor(merged_data_train[["hh21"]])) {
  merged_data_train[subset_hh20, "hh21"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh18
subset_hh18 <- !is.na(merged_data_train$hh18)
for (col in c(paste0("hh", 19:22))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_hh18, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for hh17
subset_hh17 <- merged_data_train$hh17 == 2
if (is.factor(merged_data_train[["hh18"]])) {
  merged_data_train[subset_hh17, "hh18"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh15
subset_hh15 <- !is.na(merged_data_train$hh15)
if (is.factor(merged_data_train[["hh16"]])) {
  merged_data_train[subset_hh15, "hh16"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh14
subset_hh14 <- merged_data_train$hh14 == 1
if (is.factor(merged_data_train[["hh15"]])) {
  merged_data_train[subset_hh14, "hh15"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh12 (need to handle hh12 first)
subset_hh12 <- !is.na(merged_data_train$hh12)
for (col in c(paste0("hh", 13:16))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_hh12, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for hh11
subset_hh11 <- merged_data_train$hh11 == 2
if (is.factor(merged_data_train[["hh12"]])) {
  merged_data_train[subset_hh11, "hh12"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh7
subset_hh7 <- merged_data_train$hh7 == 2 & !is.na(merged_data_train$hh7)
if (is.factor(merged_data_train[["hh8"]])) {
  merged_data_train[subset_hh7, "hh8"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh6
subset_hh6 <- merged_data_train$hh6 %in% 4:5
for (col in c(paste0("hh", 7:8))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_hh6, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for hh4
subset_hh4 <- merged_data_train$hh4 < 12
for (col in c(paste0("hh", 5:8))) {
  if (is.factor(merged_data_train[[col]])) {
    merged_data_train[subset_hh4, col] <- factor("SKIPPED")
  }
}

############################# testing data ####################################
## Handle SKIPPED value for edu64
subset_edu64 <- merged_data_test$edu64 == 2
for (col in c(paste0("edu", 65))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu64, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu61
subset_edu61 <- merged_data_test$edu61 == 2 & !is.na(merged_data_test$edu61)
for (col in c(paste0("edu", 62:63))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu61, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu59
subset_edu59 <- merged_data_test$edu59 == 2
for (col in c(paste0("edu", 60))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu59, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu57
subset_edu57 <- merged_data_test$edu57 == 2
for (col in c(paste0("edu", 58))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu57, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu54
subset_edu54 <- merged_data_test$edu54 == 2
for (col in c(paste0("edu", 55:56))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu54, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu50
subset_edu50 <- merged_data_test$edu50 == 2 & !is.na(merged_data_test$edu50)
for (col in c(paste0("edu", 51:56))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu50, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu48
subset_edu48 <- merged_data_test$edu48 == 4
for (col in c(paste0("edu", 49))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu48, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu46
subset_edu46 <- merged_data_test$edu46 == 2 & !is.na(merged_data_test$edu46)
for (col in c(paste0("edu", 47:49))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu46, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu43
subset_edu43 <- merged_data_test$edu43 == 1 & !is.na(merged_data_test$edu43)
for (col in c(paste0("edu", 44))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu43, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu30
subset_edu30 <- merged_data_test$edu30 == 2
for (col in c(paste0("edu", 31))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu30, col] <- factor("SKIPPED")
  }
}
merged_data_test[is.na(merged_data_test$edu30), "edu30"] <- 2

## Handle SKIPPED value for edu24
subset_edu24_1 <- merged_data_test$edu24 < 5 & !is.na(merged_data_test$edu24)
if (is.factor(merged_data_test[["edu25"]])) {
  merged_data_test[subset_edu24_1, "edu25"] <- factor("SKIPPED")
}

subset_edu24_2 <- merged_data_test$edu24 == 999 & !is.na(merged_data_test$edu24)
for (col in c(paste0("edu", 25:32))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu24_2, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu20
subset_edu20 <- merged_data_test$edu20 %in% 1:2
for (col in c(paste0("edu", 21:66))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu20, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu19
subset_edu19 <- merged_data_test$edu19 == 2 & !is.na(merged_data_test$edu19)
for (col in c(paste0("edu", 20:66))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu19, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu17
subset_edu17 <- merged_data_test$edu17 == 13 & !is.na(merged_data_test$edu17)
for (col in c(paste0("edu", 18:66))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu17, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu16
subset_edu16 <- merged_data_test$edu16 %in% 1:13
for (col in c(paste0("edu", 17:19))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu16, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu15
subset_edu15 <- merged_data_test$edu15 == 1 & !is.na(merged_data_test$edu15)
for (col in c(paste0("edu", 16:20))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu15, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu14
subset_edu14 <- merged_data_test$edu14 == 2 & !is.na(merged_data_test$edu14)
for (col in c(paste0("edu", 15:16))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu14, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu11
subset_edu11 <- merged_data_test$edu11 %in% 1:14
for (col in c(paste0("edu", 12:13))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu11, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu10
subset_edu10 <- merged_data_test$edu10 %in% 1:13
for (col in c(paste0("edu", 11:13))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu10, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu9
subset_edu9 <- merged_data_test$edu9 == 1 & !is.na(merged_data_test$edu9)
for (col in c(paste0("edu", 10:11))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu9, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu8
subset_edu8 <- merged_data_test$edu8 == 2 & !is.na(merged_data_test$edu8)
for (col in c(paste0("edu", 9:10))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu8, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for edu3
subset_edu3 <- merged_data_test$edu3 == 2
for (col in c(paste0("edu", 4:66))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_edu3, col] <- factor("SKIPPED")
  }
}


## Handle SKIPPED value for hh21
subset_hh21 <- !is.na(merged_data_test$hh21)
if (is.factor(merged_data_test[["hh22"]])) {
  merged_data_test[subset_hh21, "hh22"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh20
subset_hh20 <- merged_data_test$hh20 == 1
if (is.factor(merged_data_test[["hh21"]])) {
  merged_data_test[subset_hh20, "hh21"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh18
subset_hh18 <- !is.na(merged_data_test$hh18)
for (col in c(paste0("hh", 19:22))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_hh18, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for hh17
subset_hh17 <- merged_data_test$hh17 == 2
if (is.factor(merged_data_test[["hh18"]])) {
  merged_data_test[subset_hh17, "hh18"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh15
subset_hh15 <- !is.na(merged_data_test$hh15)
if (is.factor(merged_data_test[["hh16"]])) {
  merged_data_test[subset_hh15, "hh16"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh14
subset_hh14 <- merged_data_test$hh14 == 1
if (is.factor(merged_data_test[["hh15"]])) {
  merged_data_test[subset_hh14, "hh15"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh12 (need to handle hh12 first)
subset_hh12 <- !is.na(merged_data_test$hh12)
for (col in c(paste0("hh", 13:16))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_hh12, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for hh11
subset_hh11 <- merged_data_test$hh11 == 2
if (is.factor(merged_data_test[["hh12"]])) {
  merged_data_test[subset_hh11, "hh12"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh7
subset_hh7 <- merged_data_test$hh7 == 2 & !is.na(merged_data_test$hh7)
if (is.factor(merged_data_test[["hh8"]])) {
  merged_data_test[subset_hh7, "hh8"] <- factor("SKIPPED")
}

## Handle SKIPPED value for hh6
subset_hh6 <- merged_data_test$hh6 %in% 4:5
for (col in c(paste0("hh", 7:8))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_hh6, col] <- factor("SKIPPED")
  }
}

## Handle SKIPPED value for hh4
subset_hh4 <- merged_data_test$hh4 < 12
for (col in c(paste0("hh", 5:8))) {
  if (is.factor(merged_data_test[[col]])) {
    merged_data_test[subset_hh4, col] <- factor("SKIPPED")
  }
}

```

# More preprocessing numerical values
```{r}
# A function to get the mode of a data set
get_mode <- function(x) {
  uniq_x <- unique(x)
  uniq_x[which.max(tabulate(match(x, uniq_x)))] 
}
############################# training data ####################################
# edu
## Handle missing value for edu7 (mode)
edu7_mode <- get_mode(merged_data_train$edu7[!is.na(merged_data_train$edu7)])
merged_data_train$edu7[is.na(merged_data_train$edu7)] <- edu7_mode

## Handle missing value for edu18 (mode)
edu18_mode <- get_mode(merged_data_train$edu18[!is.na(merged_data_train$edu18)])
merged_data_train$edu18[is.na(merged_data_train$edu18)] <- edu18_mode

## values are skipped mostly: edu24, edu26, edu27, edu29, edu31, edu33:41, 
##   edu47, edu49, edu51, edu55, edu56, edu58, edu60, edu62, edu65, so we may
##   create new indicators to show this. When fitting, we will discard these 
##   original variables and use the new indicators
### edu41 is the sum of edu33:40, we may create an indicator using just edu41
merged_data_train$edu33 <- NULL
merged_data_train$edu34 <- NULL
merged_data_train$edu35 <- NULL
merged_data_train$edu36 <- NULL
merged_data_train$edu37 <- NULL
merged_data_train$edu38 <- NULL
merged_data_train$edu39 <- NULL
merged_data_train$edu40 <- NULL

## Handle missing values for edu24 (create indicator)
merged_data_train$edu24_skipped <- ifelse(is.na(merged_data_train$edu24), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu24 <- NULL

## Handle missing values for edu26 (create indicator)
merged_data_train$edu26_skipped <- ifelse(is.na(merged_data_train$edu26), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu26 <- NULL

## Handle missing values for edu27 (create indicator)
merged_data_train$edu27_skipped <- ifelse(is.na(merged_data_train$edu27), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu27 <- NULL

## Handle missing values for edu29 (create indicator)
merged_data_train$edu29_skipped <- ifelse(is.na(merged_data_train$edu29), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu29 <- NULL

## Handle missing values for edu31 (set some NA to 0 and create indicator)
merged_data_train$edu31_skipped <- ifelse(is.na(merged_data_train$edu31), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu31 <- NULL

## Handle missing values for edu41 (create indicator)
merged_data_train$edu41_skipped <- ifelse(is.na(merged_data_train$edu41), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu41 <- NULL

## Handle missing values for edu47 (create indicator)
merged_data_train$edu47_skipped <- ifelse(is.na(merged_data_train$edu47), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu47 <- NULL

## Handle missing values for edu49 (create indicator)
merged_data_train$edu49_skipped <- ifelse(is.na(merged_data_train$edu49), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu49 <- NULL

## Handle missing values for edu51 (create indicator)
merged_data_train$edu51_skipped <- ifelse(is.na(merged_data_train$edu51), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu51 <- NULL

## Handle missing values for edu55 (create indicator)
merged_data_train$edu55_skipped <- ifelse(is.na(merged_data_train$edu55), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu55 <- NULL

## Handle missing values for edu56 (create indicator)
merged_data_train$edu56_skipped <- ifelse(is.na(merged_data_train$edu56), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu56 <- NULL

## Handle missing values for edu58 (create indicator)
merged_data_train$edu58_skipped <- ifelse(is.na(merged_data_train$edu58), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu58 <- NULL

## Handle missing values for edu60 (create indicator)
merged_data_train$edu60_skipped <- ifelse(is.na(merged_data_train$edu60), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu60 <- NULL

## Handle missing values for edu62 (create indicator)
merged_data_train$edu62_skipped <- ifelse(is.na(merged_data_train$edu62), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu62 <- NULL

## Handle missing values for edu65 (create indicator)
merged_data_train$edu65_skipped <- ifelse(is.na(merged_data_train$edu65), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_train$edu65 <- NULL

# hh
## hh3, hh4, hh5, hh9 does not contain any missing data
## Handle missing value for hh15 (mean)
merged_data_train$hh15[is.na(merged_data_train$hh15)] <- mean(merged_data_train$hh15, na.rm = TRUE)

## Handle missing value for hh16 (mean)
merged_data_train$hh16[is.na(merged_data_train$hh16)] <- mean(merged_data_train$hh16, na.rm = TRUE)

## Handle missing value for hh21 (mean)
merged_data_train$hh21[is.na(merged_data_train$hh21)] <- mean(merged_data_train$hh21, na.rm = TRUE)

## Handle missing value for hh22 (mean)
merged_data_train$hh22[is.na(merged_data_train$hh22)] <- mean(merged_data_train$hh22, na.rm = TRUE)


############################# testing data ####################################
# edu
## Handle missing value for edu7 (mode)
edu7_mode <- get_mode(merged_data_test$edu7[!is.na(merged_data_test$edu7)])
merged_data_test$edu7[is.na(merged_data_test$edu7)] <- edu7_mode

## Handle missing value for edu18 (mode)
edu18_mode <- get_mode(merged_data_test$edu18[!is.na(merged_data_test$edu18)])
merged_data_test$edu18[is.na(merged_data_test$edu18)] <- edu18_mode

## values are skipped mostly: edu24, edu26, edu27, edu29, edu31, edu33:41, 
##   edu47, edu49, edu51, edu55, edu56, edu58, edu60, edu62, edu65, so we may
##   create new indicators to show this. When fitting, we will discard these 
##   original variables and use the new indicators
### edu41 is the sum of edu33:40, we may create an indicator using just edu41
merged_data_test$edu33 <- NULL
merged_data_test$edu34 <- NULL
merged_data_test$edu35 <- NULL
merged_data_test$edu36 <- NULL
merged_data_test$edu37 <- NULL
merged_data_test$edu38 <- NULL
merged_data_test$edu39 <- NULL
merged_data_test$edu40 <- NULL

## Handle missing values for edu24 (create indicator)
merged_data_test$edu24_skipped <- ifelse(is.na(merged_data_test$edu24), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu24 <- NULL

## Handle missing values for edu26 (create indicator)
merged_data_test$edu26_skipped <- ifelse(is.na(merged_data_test$edu26), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu26 <- NULL

## Handle missing values for edu27 (create indicator)
merged_data_test$edu27_skipped <- ifelse(is.na(merged_data_test$edu27), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu27 <- NULL

## Handle missing values for edu29 (create indicator)
merged_data_test$edu29_skipped <- ifelse(is.na(merged_data_test$edu29), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu29 <- NULL

## Handle missing values for edu31 (set some NA to 0 and create indicator)
merged_data_test$edu31_skipped <- ifelse(is.na(merged_data_test$edu31), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu31 <- NULL

## Handle missing values for edu41 (create indicator)
merged_data_test$edu41_skipped <- ifelse(is.na(merged_data_test$edu41), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu41 <- NULL

## Handle missing values for edu47 (create indicator)
merged_data_test$edu47_skipped <- ifelse(is.na(merged_data_test$edu47), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu47 <- NULL

## Handle missing values for edu49 (create indicator)
merged_data_test$edu49_skipped <- ifelse(is.na(merged_data_test$edu49), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu49 <- NULL

## Handle missing values for edu51 (create indicator)
merged_data_test$edu51_skipped <- ifelse(is.na(merged_data_test$edu51), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu51 <- NULL

## Handle missing values for edu55 (create indicator)
merged_data_test$edu55_skipped <- ifelse(is.na(merged_data_test$edu55), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu55 <- NULL

## Handle missing values for edu56 (create indicator)
merged_data_test$edu56_skipped <- ifelse(is.na(merged_data_test$edu56), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu56 <- NULL

## Handle missing values for edu58 (create indicator)
merged_data_test$edu58_skipped <- ifelse(is.na(merged_data_test$edu58), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu58 <- NULL

## Handle missing values for edu60 (create indicator)
merged_data_test$edu60_skipped <- ifelse(is.na(merged_data_test$edu60), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu60 <- NULL

## Handle missing values for edu62 (create indicator)
merged_data_test$edu62_skipped <- ifelse(is.na(merged_data_test$edu62), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu62 <- NULL

## Handle missing values for edu65 (create indicator)
merged_data_test$edu65_skipped <- ifelse(is.na(merged_data_test$edu65), 
                                         "SKIPPED", "NOT_SKIPPED")
merged_data_test$edu65 <- NULL

# hh
## hh3, hh4, hh5, hh9 does not contain any missing data
## Handle missing value for hh15 (mean)
merged_data_test$hh15[is.na(merged_data_test$hh15)] <- mean(merged_data_test$hh15, na.rm = TRUE)

## Handle missing value for hh16 (mean)
merged_data_test$hh16[is.na(merged_data_test$hh16)] <- mean(merged_data_test$hh16, na.rm = TRUE)

## Handle missing value for hh21 (mean)
merged_data_test$hh21[is.na(merged_data_test$hh21)] <- mean(merged_data_test$hh21, na.rm = TRUE)

## Handle missing value for hh22 (mean)
merged_data_test$hh22[is.na(merged_data_test$hh22)] <- mean(merged_data_test$hh22, na.rm = TRUE)
```

## Treat newly added variables as factors
```{r}
## training
merged_data_train$edu24_skipped <- factor(merged_data_train$edu24_skipped)
merged_data_train$edu26_skipped <- factor(merged_data_train$edu26_skipped)
merged_data_train$edu27_skipped <- factor(merged_data_train$edu27_skipped)
merged_data_train$edu29_skipped <- factor(merged_data_train$edu29_skipped)
merged_data_train$edu31_skipped <- factor(merged_data_train$edu31_skipped)
merged_data_train$edu41_skipped <- factor(merged_data_train$edu41_skipped)
merged_data_train$edu47_skipped <- factor(merged_data_train$edu47_skipped)
merged_data_train$edu49_skipped <- factor(merged_data_train$edu49_skipped)
merged_data_train$edu51_skipped <- factor(merged_data_train$edu51_skipped)
merged_data_train$edu55_skipped <- factor(merged_data_train$edu55_skipped)
merged_data_train$edu56_skipped <- factor(merged_data_train$edu56_skipped)
merged_data_train$edu58_skipped <- factor(merged_data_train$edu58_skipped)
merged_data_train$edu60_skipped <- factor(merged_data_train$edu60_skipped)
merged_data_train$edu62_skipped <- factor(merged_data_train$edu62_skipped)
merged_data_train$edu65_skipped <- factor(merged_data_train$edu65_skipped)

## testing
merged_data_test$edu24_skipped <- factor(merged_data_test$edu24_skipped)
merged_data_test$edu26_skipped <- factor(merged_data_test$edu26_skipped)
merged_data_test$edu27_skipped <- factor(merged_data_test$edu27_skipped)
merged_data_test$edu29_skipped <- factor(merged_data_test$edu29_skipped)
merged_data_test$edu31_skipped <- factor(merged_data_test$edu31_skipped)
merged_data_test$edu41_skipped <- factor(merged_data_test$edu41_skipped)
merged_data_test$edu47_skipped <- factor(merged_data_test$edu47_skipped)
merged_data_test$edu49_skipped <- factor(merged_data_test$edu49_skipped)
merged_data_test$edu51_skipped <- factor(merged_data_test$edu51_skipped)
merged_data_test$edu55_skipped <- factor(merged_data_test$edu55_skipped)
merged_data_test$edu56_skipped <- factor(merged_data_test$edu56_skipped)
merged_data_test$edu58_skipped <- factor(merged_data_test$edu58_skipped)
merged_data_test$edu60_skipped <- factor(merged_data_test$edu60_skipped)
merged_data_test$edu62_skipped <- factor(merged_data_test$edu62_skipped)
merged_data_test$edu65_skipped <- factor(merged_data_test$edu65_skipped)
```

# Create five folds for cross-validation
```{r}
folds <- sample(rep(1:5, length.out = nrow(merged_data_train)))
merged_data_train$fold <- folds
fold1_test <- merged_data_train[merged_data_train$fold == 1, ]
fold2_test <- merged_data_train[merged_data_train$fold == 2, ]
fold3_test <- merged_data_train[merged_data_train$fold == 3, ]
fold4_test <- merged_data_train[merged_data_train$fold == 4, ]
fold5_test <- merged_data_train[merged_data_train$fold == 5, ]

merged_data_train$fold <- NULL
fold1_test$fold <- NULL
fold2_test$fold <- NULL
fold3_test$fold <- NULL
fold4_test$fold <- NULL
fold5_test$fold <- NULL

fold1_train <- rbind(fold2_test, fold3_test, fold4_test, fold5_test)
fold2_train <- rbind(fold1_test, fold3_test, fold4_test, fold5_test)
fold3_train <- rbind(fold1_test, fold2_test, fold4_test, fold5_test)
fold4_train <- rbind(fold1_test, fold2_test, fold3_test, fold5_test)
fold5_train <- rbind(fold1_test, fold2_test, fold3_test, fold4_test)
```

# 5-Fold CV function for Random Forest
```{r}
Loss_RF <- function(data_train, data_test, formula, mtry, ntree, nodesize) {
  N <- nrow(data_test)
  original <- data_test[, c("subjective_poverty_1", "subjective_poverty_2", 
                            "subjective_poverty_3", "subjective_poverty_4",
                            "subjective_poverty_5", "subjective_poverty_6",
                            "subjective_poverty_7", "subjective_poverty_8",
                            "subjective_poverty_9", "subjective_poverty_10")]
  model <- randomForestSRC::rfsrc(formula = formula, data = data_train,
                                  mtry = mtry, ntree = ntree, nodesize = nodesize,
                                  importance = "permute")
  predicted <- as.data.frame(get.mv.predicted(predict(model, newdata = data_test), oob = TRUE))
  predicted[predicted == 0] <- 0.0000000001
  sum_yijLogpij <- sum(original * log(predicted))
  MulticlassLogarithmicLoss <- - (1/N) * sum_yijLogpij
  return(MulticlassLogarithmicLoss)
}

Five_Fold_CV_RF <- function(formula, mtry = 9, ntree = 500, nodesize = 1) {
  result <- 1/5 * (Loss_RF(fold1_train, fold1_test, formula = formula, 
                        mtry = mtry, ntree = ntree, nodesize = nodesize) + 
                   Loss_RF(fold2_train, fold2_test, formula = formula, 
                        mtry = mtry, ntree = ntree, nodesize = nodesize) + 
                   Loss_RF(fold3_train, fold3_test, formula = formula, 
                        mtry = mtry, ntree = ntree, nodesize = nodesize) + 
                   Loss_RF(fold4_train, fold4_test, formula = formula, 
                        mtry = mtry, ntree = ntree, nodesize = nodesize) + 
                   Loss_RF(fold5_train, fold5_test, formula = formula, 
                        mtry = mtry, ntree = ntree, nodesize = nodesize))
  return(result)
}
```

# Fit a full RF model
```{r}
rf_formula_full <- as.formula("Multivar(subjective_poverty_1, subjective_poverty_2,
                                        subjective_poverty_3, subjective_poverty_4,
                                        subjective_poverty_5, subjective_poverty_6,
                                        subjective_poverty_7, subjective_poverty_8,
                                        subjective_poverty_9, subjective_poverty_10,) ~
                              edu1 + edu2 + edu3 + edu4 + edu5 + edu6 + edu7 + edu8 + edu9 +
                              edu10 + edu11 + edu12 + edu13 + edu14 + edu15 + edu16 + edu17 +
                              edu18 + edu19 + edu20 + edu21 + edu22 + edu23 + edu25 +
                              edu28 + edu30 + edu32 +
                              edu42 + edu43 + edu44 + edu45 + edu46 + edu48 +
                              edu50 + edu52 + edu53 + edu54 + edu57 +
                              edu59 + edu61 + edu63 + edu64 + edu24_skipped + edu26_skipped +
                              edu27_skipped + edu29_skipped + edu31_skipped + edu41_skipped +
                              edu47_skipped + edu49_skipped + edu51_skipped + edu55_skipped +
                              edu56_skipped + edu58_skipped + edu60_skipped + edu62_skipped +
                              edu65_skipped + edu66 + hh1 + hh2 + hh3 + hh4 + hh5 + hh6 + 
                              hh7 + hh8 + hh9 + hh10 + hh11 + hh12 + hh13 + hh14 + hh15 + 
                              hh16 + hh17 + hh18 + hh19 + hh20 + hh21 + hh22")
rf_model_full <- randomForestSRC::rfsrc(formula = rf_formula_full,
                                        data = merged_data_train,
                                        importance = "permute")
# Five_Fold_CV_RF(rf_formula_full)
```

# Variable importance
```{r}
oo <- subsample(rf_model_full, verbose = FALSE)
vimpCI <- extract.subsample(oo)$var.jk.sel.Z
```

# Fit a reduced RF model
```{r}
rf_formula_reduced <- as.formula("Multivar(subjective_poverty_1, subjective_poverty_2,
                                        subjective_poverty_3, subjective_poverty_4,
                                        subjective_poverty_5, subjective_poverty_6,
                                        subjective_poverty_7, subjective_poverty_8,
                                        subjective_poverty_9, subjective_poverty_10,) ~
                              edu1 + edu2 + edu3 + edu4 + edu5 + edu6 + edu7 + edu8 + edu9 +
                              edu10 + edu11 + edu12 + edu13 + edu14 + edu15 + edu16 + edu17 +
                              edu18 + edu19 + edu20 + edu21 + edu22 + edu23 + edu25 +
                              edu28 + edu30 + edu32 +
                              edu42 + edu43 + edu44 + edu45 + edu46 + edu48 +
                              edu50 + edu52 + edu53 + edu54 + edu57 +
                              edu59 + edu61 + edu63 + edu64 + edu24_skipped + edu26_skipped +
                              edu27_skipped + edu29_skipped + edu31_skipped + edu41_skipped +
                              edu47_skipped + edu49_skipped + edu51_skipped + edu55_skipped +
                              edu56_skipped + edu58_skipped + edu60_skipped + edu62_skipped +
                              edu65_skipped + edu66 + hh1 + hh2 + hh4 + hh5 + hh6 + 
                              hh7 + hh8 + hh9 + hh10 + hh11 + hh12 + hh13 + hh14 + hh15 + 
                              hh16 + hh17 + hh18 + hh19 + hh20 + hh21 + hh22")
rf_model_reduced <- randomForestSRC::rfsrc(formula = rf_formula_reduced,
                                           data = merged_data_train,
                                           mtry = 4, ntree = 800,
                                           nodesize = 2,
                                           importance = "permute")
# Five_Fold_CV_RF(rf_formula_reduced)
```

# Tune the RF model
```{r}
# mtry_vals <- seq(4, 8)
# ntree_vals <- seq(400, 800, by = 100)
# nodesize_vals <- seq(2, 3)
# CV_result <- data.frame(mtry = numeric(), ntree = numeric(),
#                         nodesize = numeric(), cv_score = numeric())
# for (mtry in mtry_vals) {
#   for (ntree in ntree_vals) {
#     for (nodesize in nodesize_vals) {
#       cv_score <- Five_Fold_CV_RF(formula = rf_formula_full, mtry = mtry,
#                                   ntree = ntree, nodesize = nodesize)
#       CV_result <- rbind(CV_result, data.frame(mtry = mtry, ntree = ntree,
#                                                nodesize = nodesize,
#                                                cv_score = cv_score))
#     }
#   }
# }

# CV_result[which.min(CV_result$cv_score),]
#    mtry ntree nodesize cv_score
# 43    4   800        2 1.937544
```

# Final RF model
```{r}
rf_formula_final <- as.formula("Multivar(subjective_poverty_1, subjective_poverty_2,
                                         subjective_poverty_3, subjective_poverty_4,
                                         subjective_poverty_5, subjective_poverty_6,
                                         subjective_poverty_7, subjective_poverty_8,
                                         subjective_poverty_9, subjective_poverty_10,) ~
                              edu1 + edu2 + edu3 + edu4 + edu5 + edu6 + edu7 + edu8 + edu9 +
                              edu10 + edu11 + edu12 + edu13 + edu14 + edu15 + edu16 + edu17 +
                              edu18 + edu19 + edu20 + edu21 + edu22 + edu23 + edu25 +
                              edu28 + edu30 + edu32 +
                              edu42 + edu43 + edu44 + edu45 + edu46 + edu48 +
                              edu50 + edu52 + edu53 + edu54 + edu57 +
                              edu59 + edu61 + edu63 + edu64 + edu24_skipped + edu26_skipped +
                              edu27_skipped + edu29_skipped + edu31_skipped + edu41_skipped +
                              edu47_skipped + edu49_skipped + edu51_skipped + edu55_skipped +
                              edu56_skipped + edu58_skipped + edu60_skipped + edu62_skipped +
                              edu65_skipped + edu66 + hh1 + hh2 + hh4 + hh5 + hh6 + 
                              hh7 + hh8 + hh9 + hh10 + hh11 + hh12 + hh13 + hh14 + hh15 + 
                              hh16 + hh17 + hh18 + hh19 + hh20 + hh21 + hh22")
rf_model_final <- randomForestSRC::rfsrc(formula = rf_formula_final,
                                         data = merged_data_train, 
                                         mtry = 4, ntree = 800,
                                         nodesize = 2, importance = "permute")
```

# RF prediction
```{r}
preds.rfsc <- predict(rf_model_final, newdata = merged_data_test)
preds_RF <- as.data.frame(get.mv.predicted(preds.rfsc, oob = TRUE))
preds_RF$psu_hh_idcode <- merged_data_test$psu_hh_idcode
write.csv(preds_RF, "predictions.csv", row.names = FALSE)
```

# 5-Fold CV function for Support Vector Machines
```{r}
Loss_SVM <- function(data_train, data_test, gamma, degree, kernel, cost) {
  original <- data_test[, c("subjective_poverty_1", "subjective_poverty_2", 
                            "subjective_poverty_3", "subjective_poverty_4",
                            "subjective_poverty_5", "subjective_poverty_6",
                            "subjective_poverty_7", "subjective_poverty_8",
                            "subjective_poverty_9", "subjective_poverty_10")]
  N <- nrow(data_test)
  resp <- as.factor(apply(data_train[, c("subjective_poverty_1", "subjective_poverty_2",
                                         "subjective_poverty_3", "subjective_poverty_4",
                                         "subjective_poverty_5", "subjective_poverty_6",
                                         "subjective_poverty_7", "subjective_poverty_8",
                                         "subjective_poverty_9", "subjective_poverty_10")], 
                          1, function(row) which(row == 1)))
  t_no_id <- data_train[, -1]
  train <- cbind(t_no_id[1:(ncol(t_no_id) - 25)], 
                 t_no_id[(ncol(t_no_id) - 14):ncol(t_no_id)])
  train$hh3 <- NULL
  train$hh10 <- NULL
  train$hhid <- NULL
  train$edu16 <- NULL
  train$edu31_skipped <- NULL
  train$edu49_skipped <- NULL
  train$edu60_skipped <- NULL
  train <- convert_to_numeric(train)
  model <- svm(y = resp, x = train, gamma = gamma, degree = degree,
               cost = cost, kernel = kernel, probability = TRUE)
  
  te_no_id <- data_test[, -1]
  test <- cbind(te_no_id[1:(ncol(te_no_id) - 25)], 
                te_no_id[(ncol(te_no_id) - 14):ncol(te_no_id)])
  test$hh3 <- NULL
  test$hh10 <- NULL
  test$hhid <- NULL
  test$edu16 <- NULL
  test$edu31_skipped <- NULL
  test$edu49_skipped <- NULL
  test$edu60_skipped <- NULL
  test <- convert_to_numeric(test)

  preds_model <- predict(model, newdata = test, probability = TRUE)
  probs_model <- attr(preds_model, "probabilities")
  predicted <- as.data.frame(probs_model[, order(as.numeric(colnames(probs_model)))])

  predicted[predicted == 0] <- 0.0000000001
  sum_yijLogpij <- sum(original * log(predicted))
  MulticlassLogarithmicLoss <- - (1/N) * sum_yijLogpij
  return(MulticlassLogarithmicLoss)
}

Five_Fold_CV_SVM <- function(gamma = 1/74, degree = 3, kernel = "radial", cost = 1) {
  result <- 1/5 * (Loss_SVM(fold1_train, fold1_test, gamma = gamma, degree = degree, 
                            kernel = kernel, cost = cost) + 
                   Loss_SVM(fold2_train, fold2_test, gamma = gamma, degree = degree, 
                            kernel = kernel, cost = cost) + 
                   Loss_SVM(fold3_train, fold3_test, gamma = gamma, degree = degree, 
                            kernel = kernel, cost = cost) + 
                   Loss_SVM(fold4_train, fold4_test, gamma = gamma, degree = degree, 
                            kernel = kernel, cost = cost) + 
                   Loss_SVM(fold5_train, fold5_test, gamma = gamma, degree = degree, 
                            kernel = kernel, cost = cost))
  return(result)
}
```

# Tune parameters for radial kernel (degree is ignored in this case)
```{r warning=FALSE}
# cost_vals <- c(0.5, 1, 3, 5)
# gamma_vals <- c(0.01, 1/74, 0.02, 0.25)
# radial_CV <- data.frame(cost = numeric(), gamma = numeric(), cv_score = numeric())
# for (cost in cost_vals) {
#   for (gamma in gamma_vals) {
#     cv_score <- Five_Fold_CV_SVM(gamma = gamma, cost = cost, kernel = "radial")
#     radial_CV <- rbind(radial_CV, data.frame(cost = cost, gamma = gamma,
#                                              cv_score = cv_score))
#   }
# }
```

# Tune parameters for polynomial kernel
```{r warning=FALSE}
# cost_vals <- c(0.5, 1, 2)
# gamma_vals <- c(0.005, 0.01, 0.025)
# degree_vals <- c(3)
# poly_CV <- data.frame(cost = numeric(), gamma = numeric(), 
#                       degree = numeric(), cv_score = numeric())
# for (cost in cost_vals) {
#   for (gamma in gamma_vals) {
#     for (degree in degree_vals) {
#       cv_score <- Five_Fold_CV_SVM(df = merged_data_train, gamma = gamma, cost = cost, kernel = "polynomial")
#       poly_CV <- rbind(poly_CV, data.frame(cost = cost, gamma = gamma, 
#                                            degree = degree, cv_score = cv_score))
#     }
#   }
# }
```

# Final SVM model
```{r}
response <- as.factor(apply(merged_data_train[, c("subjective_poverty_1", "subjective_poverty_2",
                                                  "subjective_poverty_3", "subjective_poverty_4",
                                                  "subjective_poverty_5", "subjective_poverty_6",
                                                  "subjective_poverty_7", "subjective_poverty_8",
                                                  "subjective_poverty_9", "subjective_poverty_10")], 
                            1, function(row) which(row == 1)))
train_no_id <- merged_data_train[, -1]
SVM_train <- cbind(train_no_id[1:(ncol(train_no_id) - 25)], 
                   train_no_id[(ncol(train_no_id) - 14):ncol(train_no_id)])
SVM_train$hh3 <- NULL
SVM_train$hh10 <- NULL
SVM_train$hhid <- NULL
SVM_train$edu16 <- NULL
SVM_train$edu31_skipped <- NULL
SVM_train$edu49_skipped <- NULL
SVM_train$edu60_skipped <- NULL
SVM_train <- convert_to_numeric(SVM_train)
SVM_model <- svm(y = response, x = SVM_train, gamma = 0.01, 
                 cost = 1, kernel = "radial", probability = TRUE)
```

# Get the SVM predictions
```{r}
SVM_test <- merged_data_test[, -1]
SVM_test$hh3 <- NULL
SVM_test$hh10 <- NULL
SVM_test$hhid <- NULL
SVM_test$edu16 <- NULL
SVM_test$edu31_skipped <- NULL
SVM_test$edu49_skipped <- NULL
SVM_test$edu60_skipped <- NULL
SVM_test <- convert_to_numeric(SVM_test)

preds_SVM <- predict(SVM_model, newdata = SVM_test, 
                     probability = TRUE)
probs_SVM <- attr(preds_SVM, "probabilities")
probs_SVM_df <- as.data.frame(probs_SVM[, order(as.numeric(colnames(probs_SVM)))])
write.csv(probs_SVM_df, "predictions_SVM2.csv", row.names = FALSE)
```

# XGboost model
```{r}
d <- merged_data_train
d <- d[, -1]
train_XGboost <- cbind(d[, 1:(ncol(d)-25)], d[, (ncol(d)-14):ncol(d)])
train_XGboost$hh3 <- NULL
train_XGboost$hhid <- NULL
train_XGboost <- as.matrix(convert_to_numeric(train_XGboost))
response_XGB <- as.numeric(response) - 1

dtrain <- xgb.DMatrix(data = train_XGboost, label = response_XGB)

params <- list(
  eta = 0.1,
  objective = "multi:softprob",
  num_class = 10,
  max_depth = 4,
  subsample = 0.75,
  eval_metric = "mlogloss"
)

nrounds <- 120
xgb_model <- xgboost(params = params, data = dtrain, nrounds = nrounds,
                     early_stopping_rounds = 10, verbose=0)
```

# Predictions using XGboost
```{r}
d_test <- merged_data_test[, -1]
d_test$hh3 <- NULL
d_test$hhid <- NULL
d_test_numeric <- as.matrix(convert_to_numeric(d_test))
pred_XGboost <- predict(xgb_model, d_test_numeric, type = "prob")
pred_probs_df <- as.data.frame(matrix(pred_XGboost, ncol = 10, byrow = TRUE))
colnames(pred_probs_df) <- paste0("subjective_poverty_", 1:10)
```

# try applying stacking
```{r}
trainA_indices <- sample(seq_len(nrow(merged_data_train)), 
                         size = 0.6 * nrow(merged_data_train))
trainA <- merged_data_train[trainA_indices, ]
trainB <- merged_data_train[-trainA_indices, ]
```

# RF model
```{r}
rf_formula_finalB <- as.formula("Multivar(subjective_poverty_1, subjective_poverty_2,
                                        subjective_poverty_3, subjective_poverty_4,
                                        subjective_poverty_5, subjective_poverty_6,
                                        subjective_poverty_7, subjective_poverty_8,
                                        subjective_poverty_9, subjective_poverty_10,) ~
                              edu1 + edu2 + edu3 + edu4 + edu5 + edu6 + edu7 + edu8 + edu9 +
                              edu10 + edu11 + edu12 + edu13 + edu14 + edu15 + edu16 + edu17 +
                              edu18 + edu19 + edu20 + edu21 + edu22 + edu23 + edu25 +
                              edu28 + edu30 + edu32 +
                              edu42 + edu43 + edu44 + edu45 + edu46 + edu48 +
                              edu50 + edu52 + edu53 + edu54 + edu57 +
                              edu59 + edu61 + edu63 + edu64 + edu24_skipped + edu26_skipped +
                              edu27_skipped + edu29_skipped + edu31_skipped + edu41_skipped +
                              edu47_skipped + edu49_skipped + edu51_skipped + edu55_skipped +
                              edu56_skipped + edu58_skipped + edu60_skipped + edu62_skipped +
                              edu65_skipped + edu66 + hh1 + hh2 + hh4 + hh5 + hh6 + 
                              hh7 + hh8 + hh9 + hh10 + hh11 + hh12 + hh13 + hh14 + hh15 + 
                              hh16 + hh17 + hh18 + hh19 + hh20 + hh21 + hh22")
rf_model_finalB <- randomForestSRC::rfsrc(formula = rf_formula_finalB,
                                          data = trainA, mtry = 4, ntree = 800,
                                          nodesize = 2, importance = "permute")
```

# RF prediction
```{r}
merged_data_test_RFB <- trainB
merged_data_test_RFB$subjective_poverty_1 <- NULL
merged_data_test_RFB$subjective_poverty_2 <- NULL
merged_data_test_RFB$subjective_poverty_3 <- NULL
merged_data_test_RFB$subjective_poverty_4 <- NULL
merged_data_test_RFB$subjective_poverty_5 <- NULL
merged_data_test_RFB$subjective_poverty_6 <- NULL
merged_data_test_RFB$subjective_poverty_7 <- NULL
merged_data_test_RFB$subjective_poverty_8 <- NULL
merged_data_test_RFB$subjective_poverty_9 <- NULL
merged_data_test_RFB$subjective_poverty_10 <- NULL

preds.rfscB <- predict(rf_model_finalB, newdata = merged_data_test_RFB)
preds_RFB <- as.data.frame(get.mv.predicted(preds.rfscB, oob = TRUE))
```

# XGboost model
```{r}
responseA <- as.factor(apply(trainA[, c("subjective_poverty_1", "subjective_poverty_2",
                                                  "subjective_poverty_3", "subjective_poverty_4",
                                                  "subjective_poverty_5", "subjective_poverty_6",
                                                  "subjective_poverty_7", "subjective_poverty_8",
                                                  "subjective_poverty_9", "subjective_poverty_10")], 
                             1, function(row) which(row == 1)))
dA <- trainA
dA <- dA[, -1]
train_XGboostA <- cbind(dA[, 1:(ncol(dA)-25)], dA[, (ncol(dA)-14):ncol(dA)])
train_XGboostA$hh3 <- NULL
train_XGboostA$hhid <- NULL
train_XGboostA <- as.matrix(convert_to_numeric(train_XGboostA))
response_XGBA <- as.numeric(responseA) - 1

dtrainA <- xgb.DMatrix(data = train_XGboostA, label = response_XGBA)

paramsA <- list(
  eta = 0.1,
  objective = "multi:softprob",
  num_class = 10,
  max_depth = 4,
  subsample = 0.75,
  eval_metric = "mlogloss"
)

nroundsA <- 120
xgb_modelA <- xgboost(params = paramsA, data = dtrain, nrounds = nroundsA,
                      early_stopping_rounds = 10, verbose=0)
```

# XGboost predictions
```{r}
d_testA <- trainB[, -1]
d_testA <- cbind(d_testA[, 1:(ncol(d_testA)-25)], 
                 d_testA[, (ncol(d_testA)-14):ncol(d_testA)])
d_testA$hh3 <- NULL
d_testA$hhid <- NULL
d_test_numericA <- as.matrix(convert_to_numeric(d_testA))
pred_XGboostA <- predict(xgb_modelA, d_test_numericA, type = "prob")
pred_probs_dfA <- as.data.frame(matrix(pred_XGboostA, ncol = 10, byrow = TRUE))
colnames(pred_probs_dfA) <- paste0("subjective_poverty_", 1:10)
```

# SVM model
```{r warning=FALSE}
responseA <- as.factor(apply(trainA[, c("subjective_poverty_1", "subjective_poverty_2",
                                        "subjective_poverty_3", "subjective_poverty_4",
                                        "subjective_poverty_5", "subjective_poverty_6",
                                        "subjective_poverty_7", "subjective_poverty_8",
                                        "subjective_poverty_9", "subjective_poverty_10")], 
                             1, function(row) which(row == 1)))
train_no_idA <- trainA[, -1]
SVM_trainA <- cbind(train_no_idA[1:(ncol(train_no_idA) - 25)], 
                   train_no_idA[(ncol(train_no_idA) - 14):ncol(train_no_idA)])
SVM_trainA$hh3 <- NULL
SVM_trainA$hh10 <- NULL
SVM_trainA$hhid <- NULL
SVM_trainA$edu16 <- NULL
SVM_trainA$edu31_skipped <- NULL
SVM_trainA$edu49_skipped <- NULL
SVM_trainA$edu60_skipped <- NULL
SVM_trainA <- convert_to_numeric(SVM_trainA)
SVM_modelA <- svm(y = responseA, x = SVM_trainA, gamma = 0.01, 
                 cost = 1, kernel = "radial", probability = TRUE)
```

# SVM predictions
```{r}
SVM_testA <- trainB[, -1]
SVM_testA <- cbind(SVM_testA[, 1:(ncol(SVM_testA)-25)], 
                   SVM_testA[, (ncol(SVM_testA)-14):ncol(SVM_testA)])
SVM_testA$hh3 <- NULL
SVM_testA$hh10 <- NULL
SVM_testA$hhid <- NULL
SVM_testA$edu16 <- NULL
SVM_testA$edu31_skipped <- NULL
SVM_testA$edu49_skipped <- NULL
SVM_testA$edu60_skipped <- NULL
SVM_testA <- convert_to_numeric(SVM_testA)

preds_SVMA <- predict(SVM_modelA, newdata = SVM_testA, 
                     probability = TRUE)
probs_SVMA <- attr(preds_SVMA, "probabilities")
probs_SVM_dfA <- as.data.frame(probs_SVMA[, order(as.numeric(colnames(probs_SVMA)))])
```

# Apply stacking (SVM and XGB)
```{r}
stacked_predictions <- cbind(probs_SVM_dfA, pred_probs_dfA)
colnames(stacked_predictions) <- c("SVM_subjective_poverty_1", "SVM_subjective_poverty_2",
                                   "SVM_subjective_poverty_3", "SVM_subjective_poverty_4",
                                   "SVM_subjective_poverty_5", "SVM_subjective_poverty_6",
                                   "SVM_subjective_poverty_7", "SVM_subjective_poverty_8",
                                   "SVM_subjective_poverty_9", "SVM_subjective_poverty_10",
                                   "XGB_subjective_poverty_1", "XGB_subjective_poverty_2",
                                   "XGB_subjective_poverty_3", "XGB_subjective_poverty_4",
                                   "XGB_subjective_poverty_5", "XGB_subjective_poverty_6",
                                   "XGB_subjective_poverty_7", "XGB_subjective_poverty_8",
                                   "XGB_subjective_poverty_9", "XGB_subjective_poverty_10")
true_labels <- trainB[, (ncol(trainB) - 24) : (ncol(trainB) - 15)]
stacked_training <- cbind(true_labels, stacked_predictions)
```

# Get the final predictions from the randomForestSRC
```{r}
response_matrix <- cbind(stacked_training$subjective_poverty_1, 
                         stacked_training$subjective_poverty_2, 
                         stacked_training$subjective_poverty_3, 
                         stacked_training$subjective_poverty_4, 
                         stacked_training$subjective_poverty_5, 
                         stacked_training$subjective_poverty_6, 
                         stacked_training$subjective_poverty_7, 
                         stacked_training$subjective_poverty_8, 
                         stacked_training$subjective_poverty_9, 
                         stacked_training$subjective_poverty_10)

response_matrix <- as.data.frame(response_matrix)
response_matrix[] <- lapply(response_matrix, factor)
stacked_predictions <- as.data.frame(stacked_predictions)
stacked_predictions[] <- lapply(stacked_predictions, as.numeric)
```

# Get the meta-model from the randomForestSRC
```{r}
meta_model_final_formula <- as.formula("Multivar(subjective_poverty_1, subjective_poverty_2,
                                                 subjective_poverty_3, subjective_poverty_4,
                                                 subjective_poverty_5, subjective_poverty_6,
                                                 subjective_poverty_7, subjective_poverty_8,
                                                 subjective_poverty_9, subjective_poverty_10) ~
                              SVM_subjective_poverty_1 + SVM_subjective_poverty_2 +
                              SVM_subjective_poverty_3 + SVM_subjective_poverty_4 +
                              SVM_subjective_poverty_5 + SVM_subjective_poverty_6 +
                              SVM_subjective_poverty_7 + SVM_subjective_poverty_8 +
                              SVM_subjective_poverty_9 + SVM_subjective_poverty_10 +
                              XGB_subjective_poverty_1 + XGB_subjective_poverty_2 +
                              XGB_subjective_poverty_3 + XGB_subjective_poverty_4 +
                              XGB_subjective_poverty_5 + XGB_subjective_poverty_6 +
                              XGB_subjective_poverty_7 + XGB_subjective_poverty_8 +
                              XGB_subjective_poverty_9 + XGB_subjective_poverty_10")
meta_model_finalB <- randomForestSRC::rfsrc(formula = meta_model_final_formula,
                                            data = stacked_training, importance = "permute")
```

# Get the meta-model predictions
```{r}
original_predictions <- cbind(probs_SVM_df, pred_probs_df)
colnames(original_predictions) <- c("SVM_subjective_poverty_1", "SVM_subjective_poverty_2",
                                    "SVM_subjective_poverty_3", "SVM_subjective_poverty_4",
                                    "SVM_subjective_poverty_5", "SVM_subjective_poverty_6",
                              "SVM_subjective_poverty_7", "SVM_subjective_poverty_8",
                              "SVM_subjective_poverty_9", "SVM_subjective_poverty_10",
                              "XGB_subjective_poverty_1", "XGB_subjective_poverty_2",
                              "XGB_subjective_poverty_3", "XGB_subjective_poverty_4",
                              "XGB_subjective_poverty_5", "XGB_subjective_poverty_6",
                              "XGB_subjective_poverty_7", "XGB_subjective_poverty_8",
                              "XGB_subjective_poverty_9", "XGB_subjective_poverty_10")
preds_meta.rfsc <- predict(meta_model_finalB, newdata = original_predictions)
preds_meta_RF <- as.data.frame(get.mv.predicted(preds_meta.rfsc, oob = TRUE))
preds_meta_RF$psu_hh_idcode <- merged_data_test$psu_hh_idcode
write.csv(preds_meta_RF, "2meta_predictions.csv", row.names = FALSE)
```

# Another meta model
```{r warning=FALSE}
another_stacked_training <- cbind(stacked_training, preds_RFB)
colnames(another_stacked_training) <- c(
                              "subjective_poverty_1", "subjective_poverty_2",
                              "subjective_poverty_3", "subjective_poverty_4",
                              "subjective_poverty_5", "subjective_poverty_6",
                              "subjective_poverty_7", "subjective_poverty_8",
                              "subjective_poverty_9", "subjective_poverty_10",
                              "SVM_subjective_poverty_1", "SVM_subjective_poverty_2",
                              "SVM_subjective_poverty_3", "SVM_subjective_poverty_4",
                              "SVM_subjective_poverty_5", "SVM_subjective_poverty_6",
                              "SVM_subjective_poverty_7", "SVM_subjective_poverty_8",
                              "SVM_subjective_poverty_9", "SVM_subjective_poverty_10",
                              "XGB_subjective_poverty_1", "XGB_subjective_poverty_2",
                              "XGB_subjective_poverty_3", "XGB_subjective_poverty_4",
                              "XGB_subjective_poverty_5", "XGB_subjective_poverty_6",
                              "XGB_subjective_poverty_7", "XGB_subjective_poverty_8",
                              "XGB_subjective_poverty_9", "XGB_subjective_poverty_10",
                              "RF_subjective_poverty_1", "RF_subjective_poverty_2",
                              "RF_subjective_poverty_3", "RF_subjective_poverty_4",
                              "RF_subjective_poverty_5", "RF_subjective_poverty_6",
                              "RF_subjective_poverty_7", "RF_subjective_poverty_8",
                              "RF_subjective_poverty_9", "RF_subjective_poverty_10"
                              )

another_model_meta <- nnet::multinom(cbind(subjective_poverty_1, subjective_poverty_2,
                                 subjective_poverty_3, subjective_poverty_4,
                                 subjective_poverty_5, subjective_poverty_6,
                                 subjective_poverty_7, subjective_poverty_8,
                                 subjective_poverty_9, subjective_poverty_10) ~ ., 
                                 data = stacked_training)
```

# Another meta model prediction
```{r}
another_original_predictions <- cbind(probs_SVM_df, pred_probs_df, preds_RF[, 1:(ncol(preds_RF) - 1)])
colnames(another_original_predictions) <- c("SVM_subjective_poverty_1", "SVM_subjective_poverty_2",
                                    "SVM_subjective_poverty_3", "SVM_subjective_poverty_4",
                                    "SVM_subjective_poverty_5", "SVM_subjective_poverty_6",
                              "SVM_subjective_poverty_7", "SVM_subjective_poverty_8",
                              "SVM_subjective_poverty_9", "SVM_subjective_poverty_10",
                              "XGB_subjective_poverty_1", "XGB_subjective_poverty_2",
                              "XGB_subjective_poverty_3", "XGB_subjective_poverty_4",
                              "XGB_subjective_poverty_5", "XGB_subjective_poverty_6",
                              "XGB_subjective_poverty_7", "XGB_subjective_poverty_8",
                              "XGB_subjective_poverty_9", "XGB_subjective_poverty_10",
                              "RF_subjective_poverty_1", "RF_subjective_poverty_2",
                              "RF_subjective_poverty_3", "RF_subjective_poverty_4",
                              "RF_subjective_poverty_5", "RF_subjective_poverty_6",
                              "RF_subjective_poverty_7", "RF_subjective_poverty_8",
                              "RF_subjective_poverty_9", "RF_subjective_poverty_10")

another_model_meta_preds <- as.data.frame(predict(another_model_meta, 
                                    newdata = another_original_predictions, type = "probs"))
another_model_meta_preds$psu_hh_idcode <- merged_data_test$psu_hh_idcode
write.csv(another_model_meta_preds, "3meta_predictions.csv", row.names = FALSE)
```









