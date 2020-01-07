

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


# Drop variables with next to no correlation with Sale Price
features_to_drop <- pred_resp_cor %>% 
  filter(abs(cor) < 0.2) %>% 
  pull(feature)

train <- train %>% 
  select(-one_of(features_to_drop))


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

train <- train %>% 
  mutate(pc1 = pca_ind$coord[, 1],
         pc2 = pca_ind$coord[, 2])

