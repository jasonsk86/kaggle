

# Load libraries ----------------------------------------------------------

library(tidyverse)
library(caret)
library(randomForest)
library(scales)
library(patchwork)
library(car)

# Global setup ------------------------------------------------------------

theme_set(ggthemes::theme_economist())

# Load data ---------------------------------------------------------------

train_orig <- read_csv('data/train.csv', col_names = T)
train <- train_orig

test_orig <- read_csv('data/test.csv', col_names = T)
test <- test_orig


# Preprocessing -----------------------------------------------------------

# Here the main steps are:
  # 1. Create summary of data (unique values, NAs, class etc)
  # 2. Deal with missing values 

# > Data summary ----  
train_summary <- train %>% 
  map_dfr(~data.frame(
              unique_values =n_distinct(.),
              class =class(.),
              NAs = sum(is.na(.))
              ),
          .id = "variable")


# > Deal with NAs ----

# Lots of missing values for pool quality, miscfeature, alley and fence etc.
train_summary %>% 
  arrange(-NAs) %>% 
  head(10)


# In the data description file, NAs represent an absence of a features
# Thus, just convert these NAs to "None" (or 0 in the case for numeric features) as that in itself will act as a useful bit of information potentially when modelling

train <- train %>% 
  mutate_if(is.character, ~replace(., is.na(.), "none")) %>% 
  mutate_if(is.numeric, ~replace(., is.na(.), 0))


# EDA ---------------------------------------------------------------------

# Here the main things to look at are: 
  # 1. Correlation
  # 2. Outliers


# > Correlation ----
correlations <- train %>% 
  select_if(is.numeric) %>% 
  cor() 

# Look at correlation between predictor variables
ggcorrplot::ggcorrplot(
    correlations,
    method = "square",
    type = "lower",
    title = "Correlation coefficients between predictor variables",
    lab = F, 
    hc.order = T,
    colors = c(muted("red"), "white", muted("green"))
  )

# Correlation of features with Sale Price
pred_resp_cor <- correlations %>% 
  as.data.frame() %>% 
  rownames_to_column(var = "feature") %>% 
  select(feature, SalePrice) %>% 
  rename("cor" = SalePrice) %>% 
  filter(feature != "SalePrice")


ggplot(pred_resp_cor, aes(x = reorder(feature, cor), y = cor, fill = cor)) +
  geom_bar(stat = "identity") +
  scale_fill_gradient2(low = muted("red"), high = muted("green"), midpoint = 0) +
  coord_flip() +
  labs(title = "Correlation of features with Sale Price, the response variable")


# Dealing with correlated variables

# There are a couple of approaches to take here
  # Drop predictor variables that have low correlation with Sale Price 
  # Drop a predictor variable that is highly correlated with one another 
  # Run a PCA to reduce dimensionality 
  # Focus on tree-based methods that are less susceptible to problems around correlation
  # Use some form of penalized regression model 

# > Outliers ----

p_out <- ggplot(train, aes(y = SalePrice)) +
  geom_boxplot() +
  scale_y_continuous(labels = comma)

p_hist <- ggplot(train, aes(x = SalePrice)) +
  geom_histogram(colour = "white") +
  scale_x_continuous(labels = comma)

wrap_elements(p_hist + p_out) + ggtitle("Distribution of House Sale Prices")

# It doesn't look like there are too many exteme values, might come back to this if we notice any problems in modelling


# Feature engineering -----------------------------------------------------


# > New features ----

# add feature for if there were any remodelling or additions made to the house 
train <- train %>% 
  mutate(remodeled = ifelse(YearRemodAdd == YearBuilt, "N", "Y"))


# > Feature extraction ----

# use PCA to reduce dimensionality and created variables based on principal componts

# Principal Componets Analysis (PCA)
train_pca <- train %>% 
  select_if(is.numeric) %>% 
  select(-SalePrice) %>% 
  prcomp(scale = TRUE, center = TRUE)

# extract observation/individual principal comonents
pca_ind <- factoextra::get_pca_ind(train_pca)
pca_ind$coord

# extract variable principal compontns
pca_var <- factoextra::get_pca_var(train_pca)
pca_var$coord

# visualize loadings for first 3 principal components
pca_var$coord %>% 
  as.data.frame() %>% 
  rownames_to_column(var = "feature") %>% 
  select(feature:Dim.3) %>% 
  pivot_longer(
    cols = -feature,
    names_to = "component",
    values_to = "score"
  ) %>% 
  ggplot(aes(x = feature, y = score)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  facet_wrap(~component)

# Scree plot - ~50% of variance explained by first three principal components
factoextra::fviz_screeplot(train_pca)

# PCA variable plot
factoextra::fviz_pca_var(train_pca)

# add pc1 and pc2 into dataset 
train <- train %>% 
  mutate(pc1 = pca_ind$coord[, 1],
         pc2 = pca_ind$coord[, 2]) %>% 
  select(-SalePrice, everything(), SalePrice) # so response variable is back at most right column


# > Remove features ----

# Drop variables with next to no correlation with Sale Price
features_to_drop <- pred_resp_cor %>% 
  filter(abs(cor) < 0.2) %>% 
  pull(feature)

train <- train %>% 
  select(-one_of(features_to_drop))


# > Binning ----

# Check for any categorical variables that have very low frequencies 
# Should hopefully make models bit more robust and not overfit to niche examples

var_frequencies <- train %>% 
  select_if(is.character) %>% 
  pivot_longer(
    cols = everything()
  ) %>% 
  count(name, value) %>% 
  

# which variables have sub-attributes with low occurences
var_frequencies %>%
  filter(n < 5) %>% 
  group_by(name) %>% 
  summarize(low_atts = n_distinct(value)) %>% 
  arrange(-low_atts)

# conditon2
  # this relates to the proximity to various conditions (if more than one is present)
  # here just bin into 'normal' and 'other' to reduce cardinality of variable

train$Condition2 %>% table()

train <- train %>% 
  mutate(Condition2 = ifelse(Condition2 == "Norm", "Norm", "Other"))

# exterior1st
  # this relates to the exterior covering the house
  # here again just group anything under 5 as other 
train$Exterior1st %>% table()

exterior1st_other <- train %>% 
  count(Exterior1st) %>% 
  filter(n < 5) %>% 
  pull(Exterior1st)

train <- train %>% 
  mutate(Exterior1st = ifelse(Exterior1st %in% exterior1st_other, "Other", Exterior1st))

# RoofMatl
train$RoofMatl %>% table()

RoofMatl_other <- train %>% 
  count(RoofMatl) %>% 
  filter(n < 5) %>% 
  pull(RoofMatl)

train <- train %>% 
  mutate(RoofMatl = ifelse(RoofMatl %in% RoofMatl_other, "Other", RoofMatl))


# > Log Transform ---- 

# Create new response value that is log transform of SalePrice. This should help reduce the skewed nature of the data
# should also decrease the effect of outliers when modelling, particularly with regression techniques

train <- train %>% 
  mutate(log_SalePrice = log10(SalePrice))

p_log_hist <- ggplot(train, aes(x = log_SalePrice)) +
  geom_histogram(colour = "white") +
  labs(x = "log Sale Price")

wrap_elements(p_hist + p_log_hist) + 
  ggtitle("Distribution of Sale Price Befoe and After Log Transform") 



# Modelling ---------------------------------------------------------------


# > Multiple linear regression -------------------------------------------------------




# Random forest -----------------------------------------------------------



# Gradient boosting -------------------------------------------------------



