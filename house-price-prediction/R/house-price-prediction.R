

# Load libraries ----------------------------------------------------------

library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(scales)
library(patchwork)
library(car)
library(broom)
library(tidylog)


# Global setup ------------------------------------------------------------

theme_set(ggthemes::theme_economist())

# Load data ---------------------------------------------------------------

df_orig <- read_csv('data/train.csv', col_names = T)

df <- df_orig


# Preprocessing -----------------------------------------------------------

# Here the main steps are:
  # 1. Create summary of data (unique values, NAs, class etc)
  # 2. Deal with missing values 

# > Data summary ----  
df_summary <- df %>% 
  map_dfr(~data.frame(
              unique_values =n_distinct(.),
              class =class(.),
              NAs = sum(is.na(.))
              ),
          .id = "variable")


# > Deal with NAs ----

# Lots of missing values for pool quality, miscfeature, alley and fence etc.
df_summary %>% 
  arrange(-NAs) %>% 
  head(10)


# In the data description file, NAs represent an absence of a features
# Thus, just convert these NAs to "None" (or 0 in the case for numeric features) as that in itself will act as a useful bit of information potentially when modelling

df <- df %>% 
  mutate_if(is.character, ~replace(., is.na(.), "none")) %>% 
  mutate_if(is.numeric, ~replace(., is.na(.), 0))


# EDA ---------------------------------------------------------------------

# Here the main things to look at are: 
  # 1. Correlation
  # 2. Outliers


# > Correlation ----
correlations <- df %>% 
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

p_out <- ggplot(df, aes(y = SalePrice)) +
  geom_boxplot() +
  scale_y_continuous(labels = comma)

p_hist <- ggplot(df, aes(x = SalePrice)) +
  geom_histogram(colour = "white") +
  scale_x_continuous(labels = comma)

wrap_elements(p_hist + p_out) + ggtitle("Distribution of House Sale Prices")

# It doesn't look like there are too many exteme values, might come back to this if we notice any problems in modelling


# Feature engineering -----------------------------------------------------


# > New features ----

# add feature for if there were any remodelling or additions made to the house 
df <- df %>% 
  mutate(remodeled = ifelse(YearRemodAdd == YearBuilt, "N", "Y"))


# > Feature extraction ----

# use PCA to reduce dimensionality and created variables based on principal componts

# Principal Componets Analysis (PCA)
pca <- df %>% 
  select_if(is.numeric) %>% 
  select(-SalePrice) %>% 
  prcomp(scale = TRUE, center = TRUE)

# extract observation/individual principal comonents
pca_ind <- factoextra::get_pca_ind(pca)
pca_ind$coord

# extract variable principal compontns
pca_var <- factoextra::get_pca_var(pca)
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
df <- df %>% 
  mutate(pc1 = pca_ind$coord[, 1],
         pc2 = pca_ind$coord[, 2]) %>% 
  select(-SalePrice, everything(), SalePrice) # so response variable is back at most right column


# > Remove features ----

# Drop variables with next to no correlation with Sale Price
features_to_drop <- pred_resp_cor %>% 
  filter(abs(cor) < 0.2) %>% 
  pull(feature)

df <- df %>% 
  select(-one_of(features_to_drop))


# > Binning ----

# Check for any categorical variables that have very low frequencies 
# Should hopefully make models bit more robust and not overfit to niche examples

var_frequencies <- df %>% 
  select_if(is.character) %>% 
  pivot_longer(
    cols = everything()
  ) %>% 
  count(name, value)
  

# which variables have sub-attributes with low occurences
vars_to_bin <- var_frequencies %>%
  filter(n < 30) %>% 
  group_by(name) %>% 
  summarize(low_atts = n_distinct(value)) %>% 
  arrange(-low_atts) %>% 
  pull(name)

vars_to_bin

# function to bin variables that have low frequencie
set_to_other <- function(col, threshold) {
  
  # first summarise column by count of each value
  counts <- table(col)
  
  low <- counts[counts < threshold]
  low <- names(low)
  
  # replace values of rare categories with "Other"
  col <- ifelse(col %in% low, "Other", col)

}


df <- df %>% 
  mutate_at(vars(one_of(vars_to_bin)), 
            ~set_to_other(., 30))


# > Log Transform ---- 

# Create new response value that is log transform of SalePrice. This should help reduce the skewed nature of the data
# should also decrease the effect of outliers when modelling, particularly with regression techniques

df <- df %>% 
  mutate(log_SalePrice = log10(SalePrice))

p_log_hist <- ggplot(df, aes(x = log_SalePrice)) +
  geom_histogram(colour = "white") +
  labs(x = "log Sale Price")

wrap_elements(p_hist + p_log_hist) + 
  ggtitle("Distribution of Sale Price Befoe and After Log Transform") 




# Train / test partition --------------------------------------------------

idx <- sample(0:1, nrow(df), replace = T, prob = c(0.15, 0.85))

train <- df[idx == 1, ]
test <- df[idx == 0, ]


# Modelling ---------------------------------------------------------------


# > Multiple linear regression -------------------------------------------------------


# >> Using original Sale Price -----------------------------------------------


lm_model <- train %>% 
  select(-one_of('pc1', 'pc2', 'log_SalePrice')) %>% 
  lm(SalePrice ~ ., data = .)

lm_model_sum <- tidy(lm_model)

summary(lm_model)$r.squared
summary(lm_model)$adj.r.squared
lm_model_sum

# Using log Sale Price ----------------------------------------------------

lm_model_2 <- train %>% 
  select(-one_of('pc1', 'pc2', 'SalePrice')) %>% 
  lm(log_SalePrice ~ ., data = .)

lm_model_2_sum <- tidy(lm_model_2)

summary(lm_model_2)$r.squared
summary(lm_model_2)$adj.r.squared
lm_model_2_sum


# Using PCA ---------------------------------------------------------------

lm_model_3 <- train %>% 
  select(pc1, pc2, SalePrice) %>% 
  lm(SalePrice ~ ., data = .)

lm_model_3_sum <- tidy(lm_model_3)

summary(lm_model_3)$r.squared
summary(lm_model_3)$adj.r.squared
lm_model_3_sum

# Random forest -----------------------------------------------------------

rf_control <- trainControl(method = "none")

rf_model <- train %>% 
  select(-one_of('pc1', 'pc2', 'log_SalePrice')) %>% 
  train(SalePrice ~ ., 
        data = .,
        method = 'rf', 
        trControl = rf_control,
        ntree = 100)

summary(rf_model)

# Gradient boosting -------------------------------------------------------

xgb_control <- trainControl(method = "repeatedcv", number = 10, repeats = 2, search = "random")

xgb_model <- train %>% 
  select(-one_of('pc1', 'pc2', 'log_SalePrice')) %>% 
  train(SalePrice ~ ., 
        data = .,
        method = 'xgbTree',
        trControl = xgb_control)

summary(xgb_model)

# Predicitons -------------------------------------------------------------

lm_pred <- predict(lm_model, test)
test$lm_pred <- lm_pred

lm2_pred <- predict(lm_model_2, test)
test$lm2_pred <- 10 ^ lm2_pred

rf_pred <- predict(rf_model, test)
test$rf_pred <- rf_pred

xgb_pred <- predict(xgb_model, test)
test$xgb_pred <- xgb_pred



# Evaluation --------------------------------------------------------------

root_mean_squared_error <- function(pred) {
  sqrt(mean((test$SalePrice - pred)^2))
}

mean_absolute_error <- function(pred) {
  mean(abs(test$SalePrice - pred))
}



# summarise by root mean squared error
test %>% 
  select(contains("pred")) %>% 
  summarise_all(list(~root_mean_squared_error(.), ~mean_absolute_error(.))) %>% 
  pivot_longer(cols = everything(),
               names_to = "key",
               values_to = "value") %>% 
  mutate(key = str_replace_all(key, "pred_", ""),
         model = str_extract(key, "^[^_]*"),
         metric = str_replace_all(key, paste0(model, "_"), "")) %>% 
  ggplot() +
  geom_bar(stat = "identity", aes(x = model, y = value, fill = "firebrick")) +
  scale_y_continuous(label = scales::comma) +
  facet_wrap(~metric, scales = "free_y") +
  theme(legend.position = "none")


# lm2 appears to have the better evaluation metrics for rmse and mae. That is the model that was trained on log transform of 
# Sale Price, with predictions then converted back to the original scale

