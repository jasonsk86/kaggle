

# Load libraries ----------------------------------------------------------

library(tidyverse)
library(caret)
library(randomForest)
library(scales)


# Global setup ------------------------------------------------------------

theme_set(ggthemes::theme_fivethirtyeight())

# Load data ---------------------------------------------------------------

train_orig <- read_csv('data/train.csv', col_names = T)
train <- train_orig

test_orig <- read_csv('data/test.csv', col_names = T)
test <- test_orig


data_desc <- read_delim('data_description.txt', delim = " ")

# Preprocessing -----------------------------------------------------------

# Create a summary table of the features 
train_summary <- train %>% 
  map_dfr(~data.frame(
              unique_values =n_distinct(.),
              class =class(.),
              NAs = sum(is.na(.))
              ),
          .id = "variable")

# Lots of missing values for pool quality, miscfeature, alley and fence
train_summary %>% 
  arrange(-NAs) %>% 
  head(10)


# In the data description file, NAs represent an absence of a features
# Thus, just convert these NAs to "None" (or 0 in the case for numeric features) as that in itself will act as a useful bit of information potentially when modelling

train <- train %>% 
  mutate_if(is.character, ~replace(., is.na(.), "none")) %>% 
  mutate_if(is.numeric, ~replace(., is.na(.), 0))

# EDA ---------------------------------------------------------------------

# > Correlation


correlations <- train %>% 
  select_if(is.numeric) %>% 
  cor() 

# Look at correlation between predictor variables
ggcorrplot::ggcorrplot(
    method = "square",
    type = "lower",
    title = "Correlation coefficients between predictor variables",
    lab = F, 
    hc.order = T,
    colors = c(muted("red"), "white", muted("green"))
  )

# Correlation of features with Sale Price
correlations %>% 
  as.data.frame() %>% 
  rownames_to_column(var = "feature") %>% 
  select(feature, SalePrice) %>% 
  rename("cor" = SalePrice) %>% 
  filter(feature != "SalePrice") %>% 
  ggplot(aes(x = reorder(feature, cor), y = cor, fill = cor)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient2(low = muted("red"), high = muted("green"), midpoint = 0) +
  coord_flip() +
  labs(title = "Correlation of features with Sale Price, the response variable")

# Look at correlation 