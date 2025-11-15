# ---------- Load Packages -------------------
library(readr)
library(tidyverse)
library(ggplot2)
library(reshape2)
library(caret)
library(pROC)
library(e1071)
library(ROSE)
library(vcd) 
library(igraph)
library(ggraph)

# ---------- Import Data -------------------

teleco <- read_csv("Telco-Customer-Churn.csv", show_col_types = FALSE) 
dim(teleco)


# ---------- Preprocessing steps -------------------
str(teleco)
colSums(is.na(teleco))

#11 NAs in TotalCharges

teleco[is.na(teleco$TotalCharges), c("customerID", "tenure", "MonthlyCharges", "TotalCharges")]

# all the NAs are where tenure = 0 which means that they have not had a billing cycle yet
# Imputation for TotalCharges should be 0
teleco$TotalCharges[is.na(teleco$TotalCharges)] <- 0
colSums(is.na(teleco))

# Convert all the character cols to factors 
teleco <- teleco %>%
  mutate_if(is.character, as.factor)

#Convert dummy to factor
teleco$SeniorCitizen <- as.factor(teleco$SeniorCitizen)
levels(teleco$SeniorCitizen) <- c("No", "Yes")

str(teleco)

# Check target balance
table(teleco$Churn)
prop.table(table(teleco$Churn))

# customerID is just an identifier gives us no Predictive Value
teleco <- teleco %>% select(-customerID)



# ---------- Exploratory Data Analysis  -------------------

#Distibution of Churn
ggplot(teleco, aes(x = Churn, fill = Churn)) +
  geom_bar() + 
  theme_classic() +
  labs(title = "Churn Distribution")

# Majority "No" -> class imbalance -> possible limitation since Naive Bayes favor dominant classes
# Accuracy may look high but recall (true churn detection) will be poor.
# Possible solution -> do resampling


# Technically yes its imbalanced, but its not extreme. you would call it extreme if one class was under ~ 10 - 15%
# With that said a naive model could get ~ 73% accuracy just by predicting "no churn"
# so accuracy alone isnt meaningful focus on other metrics like F1-socre Recall and Precision

# Distribution of variables
teleco %>%
  select_if(is.factor) %>%
  pivot_longer(cols = everything(), 
               names_to = "Variable", 
               values_to = "Value") %>%
  ggplot(aes(x = Value, fill = Value)) +
  geom_bar() + 
  facet_wrap(~Variable, scales = "free") +
  theme_classic() +
  theme(legend.position = "none") +
  labs(title = "Distribution of Categorical Variables")

# Tenure
ggplot(teleco, aes(x = tenure)) +
  geom_histogram(bins = 25, fill = "skyblue", color = "white") +
  theme_minimal() +
  labs(title = "Distribution of Tenure",
       subtitle = "U-shaped — many customers leave early or stay long-term",
       x = "Tenure (months)", y = "Count")

# Monthly Charges
ggplot(teleco, aes(x = MonthlyCharges)) +
  geom_histogram(bins = 25, fill = "skyblue", color = "white") +
  theme_minimal() +
  labs(title = "Distribution of Monthly Charges",
       subtitle = "Right-skewed — many low-paying customers, fewer high-paying ones",
       x = "Monthly Charges ($)", y = "Count")

# Total Charges
ggplot(teleco, aes(x = TotalCharges)) +
  geom_histogram(bins = 25, fill = "skyblue", color = "white") +
  theme_minimal() +
  labs(title = "Distribution of Total Charges",
       subtitle = "Right-skewed — most customers have low total charges due to short tenure",
       x = "Total Charges ($)", y = "Count")


# Naive Bayes assumes independacne 
# Select numeric columns
num_data <- teleco %>% select(where(is.numeric))

# Compute correlation matrix
corr_matrix <- round(cor(num_data), 2)

# Convert to long format for ggplot
melted_corr <- melt(corr_matrix)

# Plot
ggplot(melted_corr, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = value), color = "black", size = 5) +
  scale_fill_gradient2(low = "#56B1F7", high = "#CA0020", mid = "white",
                       midpoint = 0, limit = c(-1, 1), name = "Correlation") +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1, size = 12),
    axis.text.y = element_text(size = 12),
    plot.title = element_text(face = "bold", size = 14)
  ) +
  labs(
    title = "Correlation Heatmap of Numerical Variables",
    subtitle = "Strong correlation (r = 0.83) between tenure and TotalCharges",
    x = "", y = ""
  )

# remove Total Charges 
teleco <- teleco %>% select(-TotalCharges)


# making the numeric variables into bins to have all the data as factors 
teleco <- teleco %>%
  # Bin tenure
  mutate(
    tenure_bin = cut(
      tenure,
      breaks = c(0, 12, 24, 48, 72),
      labels = c("New", "Early", "Established", "Loyal"),
      include.lowest = TRUE
    ),
    # Bin monthly charges
    MonthlyCharges_bin = cut(
      MonthlyCharges,
      breaks = 4,
      labels = c("Low", "Mid", "High", "Very High"),
      include.lowest = TRUE
    )
  ) %>%
  # Drop original continuous columns if you only want categorical ones for Naive Bayes
  select(-tenure, -MonthlyCharges)


glimpse(teleco)
table(teleco$tenure_bin)
table(teleco$MonthlyCharges_bin)

# ---- Select only categorical variables ----
cat_vars <- teleco %>% select(where(is.factor))

# ---- Function to compute Chi-square + Cramér’s V ----
chi_cramer <- function(var1, var2, data) {
  tab <- table(data[[var1]], data[[var2]])
  
  # Skip if invalid (e.g., too few unique levels)
  if (nrow(tab) < 2 | ncol(tab) < 2) return(NULL)
  
  chi <- suppressWarnings(chisq.test(tab))
  
  # Get Cramér’s V only if significant
  cramers_v <- if (chi$p.value < 0.05) {
    assocstats(tab)$cramer
  } else {
    NA
  }
  
  tibble(
    Var1 = var1,
    Var2 = var2,
    ChiSquare = chi$statistic,
    df = chi$parameter,
    p_value = chi$p.value,
    CramersV = cramers_v
  )
}

# ---- Apply to all pairs of categorical vars ----
cat_pairs <- combn(names(cat_vars), 2, simplify = FALSE)

results <- map_dfr(cat_pairs, ~chi_cramer(.x[1], .x[2], cat_vars))

# ---- Clean summary ----
results_summary <- results %>%
  arrange(p_value) %>%
  mutate(Significant = ifelse(p_value < 0.05, "Yes", "No"))

# ---- View strongest dependencies ----
results_summary %>%
  filter(Significant == "Yes") %>%
  arrange(desc(CramersV)) 

# ---- Filter to only strong dependencies ----
viz_matrix <- results_summary %>%
  filter(!is.na(CramersV), CramersV > 0.6) %>%      # keep only significant and strong
  select(Var1, Var2, CramersV)

# ---- Plot heatmap for strong associations ----
ggplot(viz_matrix, aes(x = Var1, y = Var2, fill = CramersV)) +
  geom_tile(color = "white") +
  scale_fill_gradient(low = "lightblue", high = "steelblue") +
  theme_minimal(base_size = 11) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    axis.text.y = element_text(size = 9),
    panel.grid = element_blank()
  ) +
  labs(
    title = "Strong Categorical Associations (Cramér’s V > 0.6)",
    x = "Variable 1",
    y = "Variable 2",
    fill = "Cramér’s V"
  )


# ---- Combine related service variables into higher-level features ----
teleco <- teleco %>%
  mutate(
    Streaming = ifelse(StreamingTV == "Yes" | StreamingMovies == "Yes", "Yes", "No"),
    OnlineGuardrails = ifelse(
      OnlineSecurity == "Yes" | OnlineBackup == "Yes" | TechSupport == "Yes",
      "Yes", "No"
    ),
    Streaming = factor(Streaming, levels = c("No", "Yes")),
    OnlineGuardrails = factor(OnlineGuardrails, levels = c("No", "Yes"))
  ) %>%
  select(-StreamingTV, -StreamingMovies, -OnlineSecurity, -OnlineBackup, -TechSupport, -MultipleLines)



dim(teleco)

str(teleco)

teleco <- teleco %>%
  mutate(
    PaymentMethod = case_when(
      PaymentMethod == "Electronic check" ~ "E-Check",
      PaymentMethod == "Mailed check" ~ "Mailed",
      PaymentMethod == "Bank transfer (automatic)" ~ "BankAuto",
      PaymentMethod == "Credit card (automatic)" ~ "CardAuto",
      TRUE ~ as.character(PaymentMethod)
    ),
    PaymentMethod = factor(PaymentMethod, levels = c("E-Check", "Mailed", "BankAuto", "CardAuto"))
  )

# Distribution of variables
teleco %>%
  select_if(is.factor) %>%
  pivot_longer(cols = everything(), 
               names_to = "Variable", 
               values_to = "Value") %>%
  ggplot(aes(x = Value, fill = Value)) +
  geom_bar() + 
  facet_wrap(~Variable, scales = "free") +
  theme_classic() +
  theme(legend.position = "none") +
  labs(title = "Distribution of Categorical Variables")

unique(teleco$PaymentMethod)


# ------------ Naive Bayes -------------

# for the Teleco dataset so where there are numeric variables

set.seed(123)
train_index <- createDataPartition(teleco$Churn, p = 0.8, list = FALSE)
train <- teleco[train_index, ]
test <- teleco[-train_index, ]

# Combine both sets with a new column indicating source
train$dataset <- "Train"
test$dataset  <- "Test"

# Combine into one dataframe
combined <- rbind(train, test)

# Compute proportions
churn_dist <- combined %>%
  group_by(dataset, Churn) %>%
  summarise(count = n(), .groups = 'drop') %>%
  group_by(dataset) %>%
  mutate(percent = count / sum(count) * 100)

# Plot
ggplot(churn_dist, aes(x = dataset, y = percent, fill = Churn)) +
  geom_bar(stat = "identity", position = "dodge", color = "white") +
  geom_text(aes(label = sprintf("%.1f%%", percent)),
            position = position_dodge(width = 0.9), vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("#56B1F7", "#CA0020")) +
  theme_minimal() +
  labs(
    title = "Churn Distribution in Train vs Test Sets",
    subtitle = "Proportions remain consistent — confirming stratified sampling",
    x = "Dataset",
    y = "Percentage of Customers",
    fill = "Churn"
  ) +
  theme(
    plot.title = element_text(size = 14, face = "bold"),
    plot.subtitle = element_text(size = 11)
  )

train <- train %>% select(-dataset)





# ---- Helper: train & evaluate a NB model and return metrics row ----
# Requires: e1071, caret, dplyr, pROC, purrr, tibble
eval_nb_cv <- function(train_df, test_df, label, k = 5, plot_roc = FALSE, laplace = 1) {
  set.seed(123)
  train_df <- train_df %>% dplyr::mutate(Churn = factor(Churn, levels = c("No","Yes")))
  test_df  <- test_df  %>% dplyr::mutate(Churn = factor(Churn, levels = c("No","Yes")))
  
  # Stratified folds
  folds <- caret::createFolds(train_df$Churn, k = k, returnTrain = TRUE)
  
  # Collect out-of-fold validation probabilities for threshold tuning
  val_probs  <- numeric(nrow(train_df))
  for (i in seq_along(folds)) {
    tr_idx  <- folds[[i]]
    fit_i   <- e1071::naiveBayes(Churn ~ ., data = train_df[tr_idx, , drop = FALSE], laplace = laplace)
    val_idx <- setdiff(seq_len(nrow(train_df)), tr_idx)
    val_probs[val_idx] <- predict(fit_i, train_df[val_idx, , drop = FALSE], type = "raw")[, "Yes"]
  }
  val_truth <- train_df$Churn
  
  # Search threshold using pooled OOF predictions
  thresholds <- seq(0, 1, by = 0.01)
  th_metrics <- purrr::map_dfr(thresholds, function(th) {
    pred <- factor(ifelse(val_probs >= th, "Yes", "No"), levels = c("No","Yes"))
    cm <- caret::confusionMatrix(pred, val_truth, positive = "Yes")
    tibble::tibble(threshold = th, Kappa = as.numeric(cm$overall["Kappa"]))
  })
  best_th <- th_metrics$threshold[which.max(th_metrics$Kappa)]
  
  # Refit on full training data and evaluate on the untouched test set
  fit_full  <- e1071::naiveBayes(Churn ~ ., data = train_df, laplace = laplace)
  prob_test <- predict(fit_full, test_df, type = "raw")[, "Yes"]
  pred_test <- factor(ifelse(prob_test >= best_th, "Yes", "No"), levels = c("No","Yes"))
  cm <- caret::confusionMatrix(pred_test, test_df$Churn, positive = "Yes")
  
  tp <- cm$table[2,2]; tn <- cm$table[1,1]; fp <- cm$table[1,2]; fn <- cm$table[2,1]
  precision   <- tp / (tp + fp)
  recall      <- tp / (tp + fn)
  f1          <- 2 * precision * recall / (precision + recall)
  specificity <- tn / (tn + fp)
  accuracy    <- (tp + tn) / sum(cm$table)
  bal_acc     <- (recall + specificity) / 2
  kappa       <- as.numeric(cm$overall["Kappa"])
  
  if (plot_roc) {
    roc_obj <- pROC::roc(test_df$Churn, prob_test, levels = c("No","Yes"))
    plot(roc_obj, main = sprintf("%s (AUC = %.3f)", label, pROC::auc(roc_obj)))
    abline(a = 0, b = 1, lty = 2, col = "gray")
  }
  
  tibble::tibble(
    Model         = label,
    Folds         = k,
    BestThreshold = round(best_th, 3),
    Accuracy      = round(accuracy, 3),
    Precision     = round(precision, 3),
    Recall        = round(recall, 3),
    Specificity   = round(specificity, 3),
    F1            = round(f1, 3),
    BalAccuracy   = round(bal_acc, 3),
    Kappa         = round(kappa, 3),
    TP = tp, TN = tn, FP = fp, FN = fn
  )
}


# ---- Build resampled training sets (single place) ----
mk_datasets <- function(train_df){
  # Baseline
  base <- train_df
  
  # ROSE
  rose <- ROSE(Churn ~ ., data = train_df, seed = 123)$data
  
  # Upsample (caret returns 'Class' column -> rename to 'Churn')
  up <- upSample(x = select(train_df, -Churn), y = train_df$Churn)
  names(up)[ncol(up)] <- "Churn"
  
  # Downsample
  down <- downSample(x = select(train_df, -Churn), y = train_df$Churn)
  names(down)[ncol(down)] <- "Churn"
  
  list(
    Baseline = base,
    ROSE = rose,
    Upsample = up,
    Downsample = down
  )
}


# this is for teleco not teleco_bins
resampled_trains <- mk_datasets(train)

# Train & evaluate all models (Baseline, ROSE, Upsample, Downsample)
results <- bind_rows(lapply(names(resampled_trains), function(nm){
  eval_nb_cv(resampled_trains[[nm]], test, nm)
}))


results 

library(dplyr)
library(kableExtra)
library(scales)

# -------- 1) Performance table (no Folds, no BalAccuracy) --------
perf_tbl <- results %>%
  select(Model, BestThreshold, Accuracy, Precision, Recall, Specificity, F1, Kappa) %>%
  mutate(
    BestThreshold = round(BestThreshold, 2),
    across(c(Accuracy, Precision, Recall, Specificity, F1, Kappa), ~ round(., 3))
  )

perf_tbl %>%
  kbl(
      align = "c", booktabs = TRUE) %>%
  kable_styling(full_width = FALSE,
                bootstrap_options = c("striped","hover","condensed")) %>%
  add_header_above(c("Model" = 1,
                     "Threshold" = 1,
                     "Performance Metrics" = 5,
                     "Randomness" = 1)) %>%
  column_spec(1, bold = TRUE)

# -------- 2) Confusion matrix table with colors --------
cm_raw <- results %>%
  select(Model, TP, TN, FP, FN)

# normalize each count column (for coloring)
cm_norm <- cm_raw %>%
  mutate(across(-Model, ~ scales::rescale(., to = c(0,1))))



# build a colored version using cell_spec (keep numbers visible)
cm_colored <- cm_raw
for (col in c("TP","TN","FP","FN")) {
  cm_colored[[col]] <- kableExtra::cell_spec(
    cm_raw[[col]]
  )
}

cm_colored %>%
  kbl(escape = FALSE,
      caption = "Confusion Matrix Counts",
      align = "c", booktabs = TRUE) %>%
  kable_styling(full_width = FALSE,
                bootstrap_options = c("striped","hover","condensed")) %>%
  add_header_above(c(" " = 1, "Confusion Matrix" = 4)) %>%
  column_spec(1, bold = TRUE)



# Compute OOF probabilities on train to tune threshold (no test leakage)
nb_threshold_curve <- function(train_df, label, k = 5) {
  set.seed(123)
  train_df <- train_df %>% mutate(Churn = factor(Churn, levels = c("No","Yes")))
  folds <- createFolds(train_df$Churn, k = k, returnTrain = TRUE)
  
  # OOF prob container
  oof_prob <- numeric(nrow(train_df))
  
  for (i in seq_along(folds)) {
    tr_idx  <- folds[[i]]
    val_idx <- setdiff(seq_len(nrow(train_df)), tr_idx)
    fit_i   <- naiveBayes(Churn ~ ., data = train_df[tr_idx, , drop = FALSE])
    oof_prob[val_idx] <- predict(fit_i, train_df[val_idx, , drop = FALSE], type = "raw")[, "Yes"]
  }
  
  thresholds <- seq(0, 1, by = 0.01)
  th_tbl <- map_dfr(thresholds, function(th) {
    pred <- factor(ifelse(oof_prob >= th, "Yes", "No"), levels = c("No","Yes"))
    cm   <- caret::confusionMatrix(pred, train_df$Churn, positive = "Yes")
    tibble(threshold = th, Kappa = as.numeric(cm$overall["Kappa"]))
  }) %>% mutate(Model = label)
  
  th_tbl
}

# Build curves for all resampled train sets
curves <- bind_rows(lapply(names(resampled_trains), function(nm) {
  nb_threshold_curve(resampled_trains[[nm]], nm, k = 5)
}))

# Best threshold per model (vertical guide lines)
best_lines <- curves %>%
  group_by(Model) %>%
  slice_max(Kappa, n = 1, with_ties = FALSE)

# Plot: Kappa vs Threshold
ggplot(curves, aes(x = threshold, y = Kappa, color = Model)) +
  geom_line() +
  geom_vline(data = best_lines, aes(xintercept = threshold, color = Model), linetype = "dashed") +
  labs(title = "Kappa vs Decision Threshold (OOF CV on training)",
       x = "Threshold (P[Yes])", y = "Kappa") +
  theme_minimal()




# Train NB, compute ROC/AUC, return everything needed for plotting
fit_nb_roc <- function(train_df, test_df, label) {
  fit   <- naiveBayes(Churn ~ ., data = train_df)
  probs <- predict(fit, test_df, type = "raw")[, "Yes"]
  roc_o <- roc(response = test_df$Churn, predictor = probs, levels = c("No","Yes"))
  list(label = label, model = fit, roc = roc_o, auc = as.numeric(auc(roc_o)))
}

# Plot four ROC curves in a 2x2 grid
plot_roc_grid <- function(fits_list) {
  oldpar <- par(no.readonly = TRUE); on.exit(par(oldpar))
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 1))
  for (it in fits_list) {
    plot(it$roc,
         main = paste0(it$label, " (AUC = ", sprintf("%.3f", it$auc), ")"),
         col = "#CA0020", lwd = 2)
    abline(a = 0, b = 1, lty = 2, col = "gray")
  }
}

# Create ROC objects for each resampled dataset
fits <- lapply(names(resampled_trains), function(nm) {
  fit_nb_roc(resampled_trains[[nm]], test, nm)
})

plot_roc_grid(fits) 


####----####

# Requires: e1071, dplyr, caret, pROC, purrr, tibble
nb_bootstrap_and_permutation <- function(
    train_df, test_df,
    metric = c("AUC", "Kappa"),
    B_boot = 1000,           # number of bootstrap resamples
    B_perm = 1000,           # number of permutations
    laplace = 1,             # Laplace smoothing for NB
    threshold = NULL,        # used when metric = "Kappa"; if NULL, defaults to 0.5
    seed = 123
) {
  metric <- match.arg(metric)
  set.seed(seed)
  
  # Ensure target levels are consistent
  train_df <- dplyr::mutate(train_df, Churn = factor(Churn, levels = c("No","Yes")))
  test_df  <- dplyr::mutate(test_df,  Churn = factor(Churn,  levels = c("No","Yes")))
  if (is.null(threshold)) threshold <- 0.5
  
  # --- Helpers ---
  fit_nb <- function(df) e1071::naiveBayes(Churn ~ ., data = df, laplace = laplace)
  
  predict_prob_yes <- function(fit, newdata) {
    as.numeric(predict(fit, newdata, type = "raw")[, "Yes"])
  }
  
  compute_metric <- function(prob_yes, truth, metric, threshold) {
    if (metric == "AUC") {
      roc_obj <- pROC::roc(truth, prob_yes, levels = c("No","Yes"))
      as.numeric(pROC::auc(roc_obj))
    } else { # Kappa
      pred <- factor(ifelse(prob_yes >= threshold, "Yes", "No"), levels = c("No","Yes"))
      cm <- caret::confusionMatrix(pred, truth, positive = "Yes")
      as.numeric(cm$overall["Kappa"])
    }
  }
  
  # --- Fit "true" model once on full train, evaluate on test ---
  fit_true   <- fit_nb(train_df)
  prob_true  <- predict_prob_yes(fit_true, test_df)
  true_score <- compute_metric(prob_true, test_df$Churn, metric, threshold)
  
  # --- Bootstrap: resample rows of TRAIN with replacement, fit, score on fixed TEST ---
  boot_scores <- replicate(B_boot, {
    idx <- sample.int(nrow(train_df), replace = TRUE)
    boot_train <- train_df[idx, , drop = FALSE]
    fit_b  <- fit_nb(boot_train)
    prob_b <- predict_prob_yes(fit_b, test_df)
    compute_metric(prob_b, test_df$Churn, metric, threshold)
  })
  
  boot_ci <- stats::quantile(boot_scores, c(0.025, 0.975), na.rm = TRUE)
  
  # --- Permutation: shuffle labels in TRAIN, fit, score on fixed TEST ---
  perm_scores <- replicate(B_perm, {
    perm_train <- train_df
    perm_train$Churn <- sample(perm_train$Churn)  # destroy signal
    fit_p  <- fit_nb(perm_train)
    prob_p <- predict_prob_yes(fit_p, test_df)
    compute_metric(prob_p, test_df$Churn, metric, threshold)
  })
  
  # one-sided p-value: null >= observed (for AUC/Kappa larger is better)
  p_value <- mean(perm_scores >= true_score)
  
  tibble::tibble(
    Metric          = metric,
    TrueScore       = true_score,
    BootMean        = mean(boot_scores, na.rm = TRUE),
    BootCI_L        = as.numeric(boot_ci[1]),
    BootCI_U        = as.numeric(boot_ci[2]),
    PermPValue      = p_value,
    B_boot          = B_boot,
    B_perm          = B_perm,
    Laplace         = laplace,
    ThresholdUsed   = ifelse(metric == "Kappa", threshold, NA_real_)
  ) |>
    dplyr::mutate(
      Note = dplyr::case_when(
        metric == "AUC"   ~ "Threshold-free; CI via bootstrap; p-value vs. chance via permutation.",
        metric == "Kappa" ~ "Kappa at fixed threshold; consider passing best_th from CV."
      )
    ) |>
    # Attach distributions as attributes for plotting later
    structure(boot_dist = boot_scores, perm_dist = perm_scores)
}


res_auc <- nb_bootstrap_and_permutation(
  train_df = train,
  test_df  = test,
  metric   = "AUC",
  B_boot   = 1000,
  B_perm   = 1000,
  laplace  = 1
)
res_auc

boot <- attr(res_auc, "boot_dist")
perm <- attr(res_auc, "perm_dist")
true <- res_auc$TrueScore

par(mfrow = c(1,2))

# Bootstrap distribution
hist(boot, breaks = 30, col = "skyblue", main = "Bootstrap AUC distribution",
     xlab = "AUC", xlim = c(min(boot), max(boot)))
abline(v = res_auc$BootCI_L, col = "red", lwd = 2, lty = 2)
abline(v = res_auc$BootCI_U, col = "red", lwd = 2, lty = 2)
abline(v = true, col = "darkblue", lwd = 3)
legend("topleft", legend = c("True AUC", "95% CI"), col = c("darkblue","red"), lty = c(1,2))

# Permutation null distribution
hist(perm, breaks = 30, col = "lightgray", main = "Permutation Null Distribution",
     xlab = "AUC under label shuffling")
abline(v = true, col = "darkblue", lwd = 3)
legend("topright", legend = c("Observed AUC"), col = "darkblue", lwd = 2)





