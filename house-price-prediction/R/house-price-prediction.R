

# Load libraries ----------------------------------------------------------

library(tidyverse)
library(caret)
library(randomForest)


# Load data ---------------------------------------------------------------

train_orig <- read_csv('data/train.csv', col_names = T)
train <- train_orig

test_orig <- read_csv('data/test.csv', col_names = T)
test <- test_orig



# EDA ---------------------------------------------------------------------


