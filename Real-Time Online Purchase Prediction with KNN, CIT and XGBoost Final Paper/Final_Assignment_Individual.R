# ==============================
# Libraries
# ==============================
library(readr)
library(dplyr)
library(caret)
library(pROC)
library(party)
library(xgboost)
library(SHAPforxgboost)
library(data.table)
library(PRROC)
library(pdp)
library(ggplot2)

# ==============================
# Data Loading
# ==============================
onlineShop <- read_csv("~/Documents/DSMA Masters/BLOK2 Seminars/Final Paper/online_shoppers_intention.csv")


# ==============================
# Data Preprocessing
# ==============================
str(onlineShop)

# Convert categorical variables to factors
onlineShop <- onlineShop %>%
  mutate(
    Month = factor(Month),
    VisitorType = factor(VisitorType),
    Weekend = factor(Weekend, levels = c(TRUE, FALSE)),
    OperatingSystems = factor(OperatingSystems),
    Browser = factor(Browser),
    Region = factor(Region),
    TrafficType = factor(TrafficType),
    Revenue = factor(Revenue, levels = c(TRUE, FALSE),
                     labels = c("Yes", "No")
    )
  )

table(onlineShop$Revenue)
prop.table(table(onlineShop$Revenue))
#No      Yes 
#0.8452555 0.1547445 

colSums(is.na(onlineShop)) # no NAs

# Check for duplicates
dup_rows <- duplicated(onlineShop)
sum(dup_rows)
onlineShop[dup_rows, ]
# 125 observations with same variables values. These are repeated low-activity sessions, not accidental duplicates.

# Check data sparsity
numeric_vars <- onlineShop %>% select(where(is.numeric))

zero_counts <- colSums(numeric_vars == 0)
zero_prop <- zero_counts / nrow(onlineShop)

data.frame(
  Variable = names(zero_prop),
  Zero_Proportion = zero_prop)

numeric_vars$Revenue <- onlineShop$Revenue
numeric_vars %>%
  group_by(Revenue) %>%
  summarise(across(where(is.numeric), ~ mean(. == 0)))

# ==============================
# Train / Test Split
# ==============================
set.seed(123)  # reproducibility

train_index <- createDataPartition(
  y = onlineShop$Revenue,
  p = 0.7,
  list = FALSE)

train_data <- onlineShop[train_index, ]
test_data  <- onlineShop[-train_index, ]

# Check class balance
prop.table(table(train_data$Revenue))
prop.table(table(test_data$Revenue))

# Identify numeric and factor variables
num_vars <- names(train_data)[sapply(train_data, is.numeric)]
fact_vars <- names(train_data)[sapply(train_data, is.factor)]
fact_vars <- setdiff(fact_vars, "Revenue")


# ==============================
# KNN Model
# ==============================
x_train_knn <- train_data[, num_vars]
x_test_knn  <- test_data[, num_vars]

scaler <- preProcess(
  x_train_knn,
  method = c("center", "scale"))

x_train_knn_scaled <- predict(scaler, x_train_knn) # Standardization for Euclidean distance-based KNN
x_test_knn_scaled  <- predict(scaler, x_test_knn)

y_train <- train_data$Revenue
y_test  <- test_data$Revenue

# -------- EXPLORATORY KNN -------
set.seed(123)

knn_ctrl <- trainControl( # cross- validation
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

knn_grid <- expand.grid(
  k = c(3, 5, 7, 9, 15, 25)
)

knn_model <- train(
  x = x_train_knn_scaled,
  y = y_train,
  method = "knn",
  tuneGrid = knn_grid,
  trControl = knn_ctrl,
  metric = "ROC" # optimizing for   ROC- AUC, since it evaluates ranking ability, not thresholds
)
knn_model

# Predict class labels on test set
knn_pred_class <- predict(knn_model, x_test_knn_scaled)

# Sensitivity : 0.48252  -> model caches less than 50% of actual buyers   
# Predict probabilities for the positive class
knn_pred_prob <- predict(knn_model, x_test_knn_scaled, type = "prob")

# ROC curve
knn_roc <- roc(
  response = y_test,
  predictor = knn_pred_prob[, "Yes"],
  levels = c("No", "Yes"),
  direction = "<")
# Plot ROC
plot(knn_roc, main = "ROC Curve – KNN")
# AUC value
auc(knn_roc)

# ==============================
# Conditional Inference Tree
# ==============================
set.seed(123)

ctree_model_full <- ctree(
  Revenue ~ .,
  data = train_data,
  controls = ctree_control(
    mincriterion = 0.95,   # stricter splitting
    minsplit = 100,        # avoid tiny leaves
    )
)
plot(ctree_model_full)

# more conservative, easier for interpretation
ctree_model <- ctree(
  Revenue ~ .,
  data = train_data,
  controls = ctree_control(
    mincriterion = 0.99,   # stricter splitting
    minsplit = 150,        # avoid tiny leaves
    maxdepth = 3
  )
)

plot(ctree_model)
ctree_pred_class <- predict(ctree_model, newdata = test_data, type = "response")

# Probabilities (for ROC)
ctree_pred_prob <- predict(ctree_model, newdata = test_data, type = "prob")

# Convert to matrix (safe)
ctree_pred_prob_mat <- do.call(rbind, ctree_pred_prob)
# Assign column names using the factor levels
colnames(ctree_pred_prob_mat) <- levels(y_test)

ctree_roc <- roc(
  response = y_test,
  predictor = ctree_pred_prob_mat[, "Yes"],
  levels = c("No", "Yes"),
  direction = "<"
 )

plot(ctree_roc, main = "ROC Curve – Conditional Inference Tree")
auc(ctree_roc)


# ==============================
# XGBoost
# ==============================
x_train_xgb <- model.matrix(
  Revenue ~ . - 1,
  data = train_data
)

x_test_xgb <- model.matrix(
  Revenue ~ . - 1,
  data = test_data)

# Changing outcome variable to a dummy
y_train_xgb <- ifelse(train_data$Revenue == "Yes", 1, 0)
y_test_xgb  <- ifelse(test_data$Revenue == "Yes", 1, 0)

# Buyer → 1
# Non-buyer → 0

# Handle class imbalance - class weights
scale_pos_weight <- sum(y_train_xgb == 0) / sum(y_train_xgb == 1)
scale_pos_weight

# Cross- Validation
xgb_ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 3,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Define Initial Tuning Grid (simple)
xgb_grid <- expand.grid(
  nrounds = c(100, 200),
  max_depth = c(1, 3, 5),
  eta = c(0.05, 0.1),
  gamma = c(0, 1),
  colsample_bytree = c(0.8),
  min_child_weight = c(1, 5),
  subsample = c(0.8)
)

# Train model
set.seed(123)

xgb_model <- train(
  x = x_train_xgb,
  y = factor(y_train_xgb, levels = c(1, 0), labels = c("Yes", "No")),
  method = "xgbTree",
  trControl = xgb_ctrl,
  tuneGrid = xgb_grid,
  metric = "ROC",
  scale_pos_weight = scale_pos_weight
)

xgb_model

# Predict on test set 
xgb_pred_class <- predict(xgb_model, x_test_xgb)

# Predict probabilities on test set
xgb_pred_prob <- predict(xgb_model, x_test_xgb, type = "prob")

# ROC and AUC
xgb_roc <- roc(
  response = factor(y_test_xgb, levels = c(1, 0), labels = c("Yes", "No")),
  predictor = xgb_pred_prob[, "Yes"],
  levels = c("No", "Yes"),
  direction = "<"
)

plot(xgb_roc, main = "ROC Curve – XGBoost (Test Set)")
auc(xgb_roc)

# ==============================
# Threshold Analysis
# ==============================
prob_yes <- xgb_pred_prob[, "Yes"]

thresholds <- seq(0.1, 0.9, by = 0.01)

threshold_results <- lapply(thresholds, function(t) {
  
  pred_class <- factor(
    ifelse(prob_yes >= t, "Yes", "No"),
    levels = c("Yes", "No")
  )
  
  cm <- confusionMatrix(
    data = pred_class,
    reference = factor(y_test_xgb, levels = c(1, 0), labels = c("Yes", "No")),
    positive = "Yes"
  )
  
  data.frame(
    threshold = t,
    sensitivity = cm$byClass["Sensitivity"],
    specificity = cm$byClass["Specificity"],
    precision   = cm$byClass["Pos Pred Value"],
    balanced_acc = cm$byClass["Balanced Accuracy"]
  )
})

threshold_df <- do.call(rbind, threshold_results)

#Plot Threshold
# 1. Sensitivity vs Precision
plot(
  threshold_df$threshold,
  threshold_df$sensitivity,
  type = "l",
  ylim = c(0,1),
  ylab = "Metric",
  xlab = "Threshold",
  main = "Threshold Trade-off: Sensitivity vs Precision"
)

lines(threshold_df$threshold, threshold_df$precision, col = "blue")
legend("topright", legend = c("Sensitivity", "Precision"),
       col = c("black", "blue"), lty = 1)

# 2. Sensitivity vs Specificity
plot(
  threshold_df$threshold,
  threshold_df$sensitivity,
  type = "l",
  ylim = c(0,1),
  ylab = "Metric",
  xlab = "Threshold",
  main = "Sensitivity–Specificity Trade-off"
)

lines(threshold_df$threshold, threshold_df$specificity, col = "red")
legend("topright", legend = c("Sensitivity", "Specificity"),
       col = c("black", "red"), lty = 1)

pr <- pr.curve(
  scores.class0 = xgb_pred_prob[, "Yes"][y_test_xgb == 1],
  scores.class1 = xgb_pred_prob[, "Yes"][y_test_xgb == 0],
  curve = TRUE
)

plot(pr, main = "Precision–Recall Curve – XGBoost")
pr$auc.integral

# ==============================
# Global Variable Importance for XGBoost 
# ==============================
xgb_final <- xgb_model$finalModel

# Feature importance using GAIN
importance_matrix <- xgb.importance(
  model = xgb_final,
  feature_names = colnames(x_train_xgb))

# View top features
head(importance_matrix, 15)

# Plot importance
xgb.plot.importance(
  importance_matrix,
  top_n = 15,
  measure = "Gain",
  rel_to_first = TRUE,
  xlab = "Relative Gain")

# ==============================
# SHAP Analysis
# ==============================

xgb_final <- xgb_model$finalModel
# Use the SAME matrix used for training
X_shap <- as.data.frame(x_train_xgb)

# Ensure everything is numeric
X_shap[] <- lapply(X_shap, as.numeric)

# Double check
stopifnot(all(sapply(X_shap, is.numeric)))

shap_values <- shap.values(
  xgb_model = xgb_final,
  X_train = X_shap
)
str(shap_values)

shap.plot.summary(
  shap_values$shap_score,
  X_shap
)

# Convert SHAP matrix to long format
shap_long <- as.data.table(shap_values$shap_score)
shap_long[, row_id := .I]

shap_long <- melt(
  shap_long,
  id.vars = "row_id",
  variable.name = "Feature",
  value.name = "SHAP_value"
)

# Add feature values (for color)
X_long <- as.data.table(X_shap)
X_long[, row_id := .I]

X_long <- melt(
  X_long,
  id.vars = "row_id",
  variable.name = "Feature",
  value.name = "Feature_value"
)

# Merge SHAP values with feature values
shap_long <- merge(
  shap_long,
  X_long,
  by = c("row_id", "Feature")
)

# Compute mean absolute SHAP per feature
top_features <- shap_long[
  , .(mean_abs_shap = mean(abs(SHAP_value))),
  by = Feature
][order(-mean_abs_shap)]

# Keep top 15 features
top_n <- 15
top_features <- top_features[1:top_n, Feature]

shap_long <- shap_long[Feature %in% top_features]

# Order factors for plotting
shap_long[, Feature := factor(
  Feature,
  levels = rev(top_features)
)]

ggplot(
  shap_long,
  aes(
    x = SHAP_value,
    y = Feature,
    color = Feature_value
  )
) +
  geom_point(
    alpha = 0.6,
    size = 0.8
  ) +
  scale_color_gradient(
    low = "blue",
    high = "red"
  ) +
  geom_vline(
    xintercept = 0,
    linetype = "dashed",
    color = "grey40"
  ) +
  labs(
    title = "SHAP Summary Plot – XGBoost (With PageValues)",
    x = "SHAP value (impact on model output)",
    y = "Feature",
    color = "Feature value"
  ) +
  theme_minimal(base_size = 12)

# ==============================
# XGBoost without PageValues
# ==============================
set.seed(123)
train_data_noPV <- train_data %>% select(-PageValues)
test_data_noPV  <- test_data %>% select(-PageValues)

x_train_xgb_noPV <- model.matrix(
  Revenue ~ . - 1,
  data = train_data_noPV
)

x_test_xgb_noPV <- model.matrix(
  Revenue ~ . - 1,
  data = test_data_noPV
)

y_train_xgb <- ifelse(train_data$Revenue == "Yes", 1, 0)
y_test_xgb  <- ifelse(test_data$Revenue == "Yes", 1, 0)

set.seed(123)

xgb_model_noPV <- train(
  x = x_train_xgb_noPV,
  y = factor(y_train_xgb, levels = c(1, 0), labels = c("Yes", "No")),
  method = "xgbTree",
  trControl = xgb_ctrl,
  tuneGrid = xgb_grid,
  metric = "ROC",
  scale_pos_weight = scale_pos_weight
)

xgb_model_noPV

# Class prediction (threshold = 0.50)
xgb_noPV_pred_class <- predict(xgb_model_noPV, x_test_xgb_noPV)

cm_noPV <- confusionMatrix(
  xgb_noPV_pred_class,
  factor(y_test_xgb, levels = c(1, 0), labels = c("Yes", "No")),
  positive = "Yes"
)

cm_noPV

xgb_noPV_pred_prob <- predict(
  xgb_model_noPV,
  x_test_xgb_noPV,
  type = "prob"
)

xgb_noPV_roc <- roc(
  response = factor(y_test_xgb, levels = c(1, 0), labels = c("Yes", "No")),
  predictor = xgb_noPV_pred_prob[, "Yes"],
  levels = c("No", "Yes"),
  direction = "<"
)

plot(xgb_noPV_roc, main = "ROC Curve – XGBoost (No PageValues)")
auc(xgb_noPV_roc)

# ==============================
# Global Variable Importance for XGBoost without PageValues
# ==============================
xgb_final_noPV <- xgb_model_noPV$finalModel

# Convert training matrix to data.frame
X_shap_noPV <- as.data.frame(x_train_xgb_noPV)

shap_values_noPV <- shap.values(
  xgb_model = xgb_final_noPV,
  X_train = X_shap_noPV
)

shap.plot.summary(
  shap_values_noPV$shap_score,
  X_shap_noPV
)

str(shap_values_noPV)

shap_imp_noPV <- data.frame(
  Feature = names(shap_values_noPV$mean_shap_score),
  MeanAbsSHAP = shap_values_noPV$mean_shap_score
)

shap_imp_noPV <- shap_imp_noPV[order(-shap_imp_noPV$MeanAbsSHAP), ]

head(shap_imp_noPV, 15)

# ==============================
# SHAP Analysis - no PageValue
# ==============================
shap_long <- as.data.table(shap_values_noPV$shap_score)
shap_long[, id := .I]

shap_long <- melt(
  shap_long,
  id.vars = "id",
  variable.name = "Feature",
  value.name = "SHAP"
)

# Feature values (same rows!)
X_long <- as.data.table(X_shap_noPV)
X_long[, id := .I]

X_long <- melt(
  X_long,
  id.vars = "id",
  variable.name = "Feature",
  value.name = "Value"
)

# Merge SHAP + feature values
shap_plot_df <- merge(
  shap_long,
  X_long,
  by = c("id", "Feature")
)

top_features <- shap_imp_noPV$Feature[1:15]

shap_plot_df <- shap_plot_df[
  Feature %in% top_features
]

ggplot(
  shap_plot_df,
  aes(
    x = SHAP,
    y = reorder(Feature, abs(SHAP), FUN = mean),
    color = Value
  )
) +
  geom_point(
    alpha = 0.4,
    size = 1.2
  ) +
  scale_color_gradient(
    low = "blue",
    high = "red"
  ) +
  labs(
    title = "SHAP Summary Plot – XGBoost (No PageValues)",
    x = "SHAP value (impact on model output)",
    y = NULL,
    color = "Feature value"
  ) +
  theme_minimal()

# ==============================
# Final Comparison Table 
# ==============================
extract_metrics <- function(cm, auc_value) {
  data.frame(
    ROC_AUC = as.numeric(auc_value),
    Sensitivity = unname(cm$byClass["Sensitivity"]),
    Specificity = unname(cm$byClass["Specificity"]),
    Precision = unname(cm$byClass["Pos Pred Value"]),
    F1 = unname(cm$byClass["F1"]),
    Balanced_Accuracy = unname(cm$byClass["Balanced Accuracy"]),
    row.names = NULL
  )
}

# --- KNN ---
cm_knn <- confusionMatrix(knn_pred_class, y_test, positive = "Yes")
metrics_knn <- extract_metrics(cm_knn, auc(knn_roc))
metrics_knn$Model <- "KNN"

# --- Conditional Inference Tree ---
cm_ctree <- confusionMatrix(ctree_pred_class, y_test, positive = "Yes")
metrics_ctree <- extract_metrics(cm_ctree, auc(ctree_roc))
metrics_ctree$Model <- "Conditional Inference Tree"

# --- XGBoost (threshold = 0.50) ---
# Probability vector
prob_yes <- xgb_pred_prob[, "Yes"]

# Function to compute CM at a given threshold
cm_at_threshold <- function(t) {
  pred <- factor(
    ifelse(prob_yes >= t, "Yes", "No"),
    levels = c("Yes", "No")
  )
  
  confusionMatrix(
    pred,
    factor(y_test_xgb, levels = c(1,0), labels = c("Yes","No")),
    positive = "Yes"
  )
}

# --- XGBoost: High Recall ---
cm_xgb_low <- cm_at_threshold(0.20)
metrics_xgb_low <- extract_metrics(cm_xgb_low, auc(xgb_roc))
metrics_xgb_low$Model <- "XGBoost (Recall-focused, t = 0.20)"

# --- XGBoost: Balanced ---
cm_xgb_mid <- cm_at_threshold(0.50)
metrics_xgb_mid <- extract_metrics(cm_xgb_mid, auc(xgb_roc))
metrics_xgb_mid$Model <- "XGBoost (Balanced, t = 0.50)"

# --- XGBoost: High Precision ---
cm_xgb_high <- cm_at_threshold(0.70)
metrics_xgb_high <- extract_metrics(cm_xgb_high, auc(xgb_roc))
metrics_xgb_high$Model <- "XGBoost (Precision-focused, t = 0.70)"


final_results <- rbind(
  cbind(Model = "KNN",
        extract_metrics(cm_knn, auc(knn_roc))),
  
  cbind(Model = "Conditional Inference Tree",
        extract_metrics(cm_ctree, auc(ctree_roc))),
  
  cbind(Model = "XGBoost (High Recall, t = 0.20)",
        extract_metrics(cm_xgb_low, auc(xgb_roc))),
  
  cbind(Model = "XGBoost (Balanced, t = 0.50)",
        extract_metrics(cm_xgb_mid, auc(xgb_roc))),
  
  cbind(Model = "XGBoost (High Precision, t = 0.70)",
        extract_metrics(cm_xgb_high, auc(xgb_roc))),
  
  cbind(Model = "XGBoost (No PageValues, t = 0.50)",
        extract_metrics(cm_noPV, auc(xgb_noPV_roc)))
)

final_results

# ==============================
# Partial Dependence Plots
# ==============================
# PDP 1: ExitRates
pdp_exit <- partial(
  object = xgb_model_noPV,
  pred.var = "ExitRates",
  train = as.data.frame(x_train_xgb_noPV),
  prob = TRUE,
  which.class = "Yes"
)

autoplot(pdp_exit) +
  labs(
    title = "Partial Dependence Plot: ExitRates",
    x = "Exit Rate",
    y = "Predicted Probability of Purchase"
  ) +
  theme_minimal()

# PDP 2: ProductRelated_Duration
pdp_prod_dur <- partial(
  object = xgb_model_noPV,
  pred.var = "ProductRelated_Duration",
  train = as.data.frame(x_train_xgb_noPV),
  prob = TRUE,
  which.class = "Yes"
)

autoplot(
  pdp_prod_dur
) +
  labs(
    title = "Partial Dependence Plot: ProductRelated Duration",
    x = "Time Spent on Product Pages",
    y = "Predicted Probability of Purchase"
  ) +
  theme_minimal()

# PDP 3: BounceRates
pdp_bounce <- partial(
  object = xgb_model_noPV,
  pred.var = "BounceRates",
  train = as.data.frame(x_train_xgb_noPV),
  prob = TRUE,
  which.class = "Yes"
)

autoplot(
  pdp_bounce
) +
  labs(
    title = "Partial Dependence Plot: BounceRates",
    x = "Bounce Rate",
    y = "Predicted Probability of Purchase"
  ) +
  theme_minimal()


