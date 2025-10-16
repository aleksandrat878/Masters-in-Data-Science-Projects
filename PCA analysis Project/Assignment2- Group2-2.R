## ----echo = FALSE, eval = FALSE, warning = FALSE, message = FALSE-------------
# # Directory setup
# path = dirname(rstudioapi::getSourceEditorContext()$path) # Path is directory of this file
# setwd(path)                                               # Set working directory


## ----include=FALSE------------------------------------------------------------
# --- Group Assignment 2 ------------------------------------------------------

# --- Libraries ---------------------------------------------------------------
library(dplyr)
library(tidyr)
library(ggplot2)
library(factoextra)
library(pls)
library(boot) 
library(paran)
library(stargazer)
set.seed(123)

# --- Load & prepare data -----------------------------------------------------
data <- read.csv("a2_data_group_2.csv")


data <- data[,-1]

names(data) <- c(
  "Date",                        
  "mean_temp",                   
  "Tmax",                        
  "tmin",                          
  "perceived_mean",              
  "perceived_max",                 
  "perceived_min",                
  "max_wind_speed",               
  "max_wind_gusts",  
  "shortwave_radiation", 
  "dominant_wind_direction",      
  "reference_evapotranspiration", 
  "daylight_duration",            
  "sunshine_duration",            
  "precipitation_sum",           
  "snowfall_sum",                 
  "precipitation_hours",          
  "rain_sum"                       
)

names(data)[names(data) == "Max..temperature...C."] <- "Tmax"
data$Date <- as.Date(data$Date, format = "%Y-%m-%d")

# --- Task 1: Build independent (X) and dependent (y) variables ---------------
# X(t) uses ALL variables except the last row; y(t+1) is Tmax excluding the first row
X      <- data[-nrow(data), ]   # all but last row #-1
y      <- data$Tmax[-1]         # Tmax shifted one day ahead
dates  <- data$Date[-1]         # matching dates for y

# Keep numeric predictors for later steps (+ drop zero-variance)
X_num <- X[sapply(X, is.numeric)]
X_num <- X_num[, sapply(X_num, function(col) sd(col, na.rm = TRUE) > 0)]
y_num  <- as.numeric(y)

# --- Task 2: Exploratory analysis -------------------------------------------
# Scatterplots of each independent variable vs. next-day Tmax
df_plot <- cbind(X, y) %>% dplyr::select(where(is.numeric))
df_plot %>%
  pivot_longer(-y, names_to = "variable", values_to = "value") %>%
  ggplot(aes(value, y)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  facet_wrap(~ variable, scales = "free", ncol = 4) +
  labs(x = "Independent variable", y = "Next-day Tmax",
       title = "Scatterplots: IVs vs next-day Tmax")

# Compute correlations
correlations <- sapply(X_num, function(col) cor(col, y_num, use = "pairwise.complete.obs"))
round(sort(correlations, decreasing = TRUE), 2)

# Convert to a data frame for ggplot
cor_df <- data.frame(
  Variable = names(correlations),
  Correlation = as.numeric(correlations)
)

# Plot
ggplot(cor_df, aes(x = reorder(Variable, Correlation), y = Correlation)) +
  geom_col(fill = "skyblue") +
  coord_flip() +
  theme_minimal() +
  labs(
    title = "Correlation of Independent Variables with Next-day Max Temperature",
    x = "Variable",
    y = "Correlation coefficient"
  )

# --- Task 3: Train-Test Split (Time Series) ---------------------------------
# Chronological split (no shuffling): 80% train, 20% test
df <- data.frame(Date = dates, X_num, y = y_num)
n  <- nrow(df); train_size <- floor(0.8 * n)
train <- df[1:train_size, ]      # earlier observations
test  <- df[(train_size + 1):n, ] # later observations

# Visual check of the split (red dashed line = boundary)
ggplot(df, aes(Date, y)) +
  geom_line(color = "steelblue") +
  geom_vline(xintercept = as.numeric(max(train$Date)),
             linetype = "dashed", color = "red") +
  labs(title = "Train/Test Split Check",
       subtitle = "Red dashed line = boundary between training and test sets",
       y = "Next-day Tmax")

# --- Task 4: Principal Component Analysis (PCA) ------------------------------
# PCA on TRAINING independent variables only (no Date, no y)
X_train <- train[, sapply(train, is.numeric)]
X_train <- X_train[, setdiff(names(X_train), "y")]

# Run PCA using correlation matrix (standardizes variables)
res <- princomp(X_train, cor = TRUE, scores = TRUE)
summary(res)

# Scree plot + quick look at loadings (PC1–PC2)
screeplot(res, main = "Scree Plot - PCA on Training Data")
round(res$loadings[, 1:4], 2)

#Parallel Analysis
paran(X_train, graph = TRUE)
##### comment #####

# Cleaner biplots (variables only + biplot with top contributors)
fviz_pca_var(res, repel = TRUE, col.var = "contrib")
fviz_pca_biplot(res,
                geom.ind = "point", pointshape = 16, pointsize = 0.7, alpha.ind = 0.15,
                col.ind = "grey70", col.var = "firebrick", repel = TRUE,
                select.var = list(contrib = 12)  # label top 12 vars
)

# Apply Kaiser’s rule (keep eigenvalues > 1)
eigenvalues <- res$sdev^2
kaiser_components <- sum(eigenvalues > 1)
cat("Number of components with eigenvalue > 1:", kaiser_components, "\n")

# --- Task 7: Biplot and Interpretation of Principal Components ---------------
loadings_df <-round(res$loadings[, 1:4], 2)

# --- Task 8 (Even group)
# 1) Point estimate (train data)
pc1_var <- (res$sdev[1]^2) / sum(res$sdev^2)
pc1_var  # ≈ 0.48

# 2) Moving-block bootstrap CI for PC1 proportion
stat_pc1 <- function(x, i) {
  xb <- x[i, , drop = FALSE]
  p  <- try(princomp(xb, cor = TRUE), silent = TRUE)
  if (inherits(p, "try-error")) return(NA_real_)
  (p$sdev[1]^2) / sum(p$sdev^2)
}
L <- 14  # block length (~2 weeks). Robust for L in {7, 14, 21}
B <- 1000
bt <- tsboot(tseries = as.matrix(X_train), statistic = stat_pc1,
             R = B, l = L, sim = "fixed")
pc1_ci_block <- quantile(bt$t[is.finite(bt$t)], c(0.025, 0.975), na.rm = TRUE)
pc1_ci_block

# 3) Variables best explained by PC1 (squared correlation between PC1 and vars)
corpca <- cor(X_train,res$scores)[,1]
explained_by_pc1 <- sort(corpca^2, decreasing=TRUE)
head(explained_by_pc1,10)


# --- Task 9: Principal Component Regression (PCR) ----------------------------
# Time-aware CV; compare CV-selected ncomp vs fixed k = 4 from Task 6
train_pcr <- subset(train, select = -Date)
test_pcr  <- subset(test,  select = -Date)

pcr_cv <- pcr(y ~ ., data = train_pcr, scale = TRUE,
              validation = "CV", segments = 10, segment.type = "consecutive")
validationplot(pcr_cv, val.type = "RMSEP")
n_cv <- selectNcomp(pcr_cv, method = "onesigma", plot = FALSE)

pcr_final_cv <- pcr(y ~ ., data = train_pcr, scale = TRUE, ncomp = n_cv)
pcr_final_k  <- pcr(y ~ ., data = train_pcr, scale = TRUE, ncomp = 4)

pred_cv <- as.numeric(predict(pcr_final_cv, newdata = test_pcr, ncomp = n_cv))
pred_k  <- as.numeric(predict(pcr_final_k,  newdata = test_pcr, ncomp = 4))

rmse <- function(e) sqrt(mean(e^2))
mae  <- function(e) mean(abs(e))
r2   <- function(y, yhat) 1 - sum((y - yhat)^2) / sum((y - mean(y))^2)

y_test <- test$y
metrics <- list(
  CV_ncomp = n_cv,
  CV_RMSE  = rmse(y_test - pred_cv),
  CV_MAE   = mae(y_test - pred_cv),
  CV_R2    = r2(y_test, pred_cv),
  K_used   = 4,
  K_RMSE   = rmse(y_test - pred_k),
  K_MAE    = mae(y_test - pred_k),
  K_R2     = r2(y_test, pred_k)
)
metrics

# --- Task 10: Benchmark Multiple Linear Regression (MLR) --------------------
lm_model <- lm(y ~ ., data = train_pcr)
summary(lm_model)
pred_lm  <- predict(lm_model, newdata = test_pcr)

lm_metrics <- list(
  RMSE = rmse(y_test - pred_lm),
  MAE  = mae(y_test - pred_lm),
  R2   = r2(y_test, pred_lm)
)
lm_metrics

# --- Task 11: Model Comparison on Test Data ----------------------------------
comparison <- data.frame(
  Model = c("PCA (4 PCs)", "Multiple Linear Regression"),
  RMSE  = c(metrics$K_RMSE, lm_metrics$RMSE),
  MAE   = c(metrics$K_MAE,  lm_metrics$MAE),
  R2    = c(metrics$K_R2,   lm_metrics$R2)
)
comparison

comparison_long <- comparison %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")
ggplot(comparison_long, aes(x = Metric, y = Value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "PCA vs. Multiple Linear Regression on Test Data",
       subtitle = "Performance comparison on held-out period",
       y = "Metric value") +
  theme_minimal()

# --- Task 12: Sensitivity Analysis of PCR to Number of Components ------------
k_values <- c(3, 4, 5)
pcr_sensitivity <- data.frame()
for (k in k_values) {
  model <- pcr(y ~ ., data = train_pcr, scale = TRUE, ncomp = k)
  preds  <- as.numeric(predict(model, newdata = test_pcr, ncomp = k))
  rmse_k <- rmse(y_test - preds)
  mae_k  <- mae(y_test - preds)
  r2_k   <- r2(y_test, preds)
  
  pcr_sensitivity <- rbind(pcr_sensitivity,
                           data.frame(
                             Components = k,
                             RMSE = rmse_k,
                             MAE = mae_k,
                             R2  = r2_k
                           ))
}

pcr_sensitivity

# Visualize the sensitivity
ggplot(pcr_sensitivity, aes(x = Components, y = RMSE)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_point(size = 3, color = "firebrick") +
  labs(
    title = "PCR Sensitivity to Number of Components",
    subtitle = "RMSE on test data for k = 3, 4, 5",
    y = "Test RMSE"
  ) +
  expand_limits(y = c(min(pcr_sensitivity$RMSE) - 0.05,
                      max(pcr_sensitivity$RMSE) + 0.05)) +
  theme_minimal()



## ----include=FALSE------------------------------------------------------------
# --- Task 13: Performance Across Temperature Levels --------------------------

# 1) Create decile groups based on true Tmax in test data
test_results <- data.frame(
  True = y_test,
  PCR_4 = pred_k,
  MLR = pred_lm
)

test_results <- test_results %>%
  mutate(Decile = ntile(True, 10))  # 10 roughly equal groups (10% each)

# 2) Compute RMSE per decile for both models
rmse <- function(e) sqrt(mean(e^2))

group_metrics <- test_results %>%
  group_by(Decile) %>%
  summarise(
    RMSE_PCR = rmse(True - PCR_4),
    RMSE_MLR = rmse(True - MLR),
    Mean_Temp = mean(True),
    .groups = "drop"   
  ) %>%
  pivot_longer(cols = starts_with("RMSE"), names_to = "Model", values_to = "RMSE")

# 3) Plot RMSE by temperature group
ggplot(group_metrics, aes(x = Decile, y = RMSE, color = Model, group = Model)) +  #### !!!!! ISSUE WITH DECILE!!!!
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_color_manual(values = c("firebrick", "steelblue"),
                     labels = c("PCR (4 PCs)", "Multiple Linear Regression")) +
  labs(
    title = "Model Performance by Temperature Decile (Test Data)",
    subtitle = "RMSE across temperature ranges (lower = better)",
    x = "Temperature decile (1 = coldest, 10 = hottest)",
    y = "RMSE"
  ) +
  theme_minimal() +
  theme(legend.title = element_blank())


## ----echo=FALSE, message=FALSE, warning=FALSE, results='asis'-----------------

# Prepare clean model comparison table
comparison_df <- data.frame(
  Model = c("PCA (4 PCs)", "Multiple Linear Regression"),
  RMSE  = round(c(metrics$K_RMSE, lm_metrics$RMSE), 3),
  MAE   = round(c(metrics$K_MAE,  lm_metrics$MAE), 3),
  R2    = round(c(metrics$K_R2,   lm_metrics$R2), 3),
  stringsAsFactors = FALSE
)

# Remove any row names to avoid duplication
rownames(comparison_df) <- NULL

# Generate LaTeX table (simple, clean style)
stargazer(
  comparison_df,
  summary = FALSE,
  title = "Comparison of PCR and MLR model performance on test data.",
  label = "tab:model_comparison",
  digits = 3,
  font.size = "small",
  float.env = "table",
  type = "latex",
  header = FALSE,
  table.placement = "!h",
  align = FALSE
)



## ----echo=FALSE, message=FALSE, warning=FALSE, results='asis'-----------------
ggplot(group_metrics, aes(x = Decile, y = RMSE, color = Model, group = Model)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2) +
  scale_color_manual(values = c("firebrick", "steelblue"),
                     labels = c("PCR (4 PCs)", "Multiple Linear Regression")) +
  labs(
    title = "Model Performance by Temperature Decile (Test Data)",
    subtitle = "RMSE across temperature ranges (lower = better)",
    x = "Temperature decile (1 = coldest, 10 = hottest)",
    y = "RMSE"
  ) +
  theme_minimal() +
  theme(legend.title = element_blank())


## ----echo=FALSE, warning=FALSE, message=FALSE, results='asis'-----------------
# ------------THE CODE BELOW IS THE CODE FOR THE TABLES INCLUDED IN THE APPENDIX -------------------------
# Scatterplots of each independent variable vs. next-day Tmax
df_plot <- cbind(X, y) %>% dplyr::select(where(is.numeric))
df_plot %>%
  pivot_longer(-y, names_to = "variable", values_to = "value") %>%
  ggplot(aes(value, y)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  geom_smooth(method = "lm", se = FALSE, color = "red") +
  facet_wrap(~ variable, scales = "free", ncol = 4) +
  labs(x = "Independent variable", y = "Next-day Tmax",
       title = "Scatterplots: IVs vs next-day Tmax")


## ----echo=FALSE, warning=FALSE, message=FALSE, results='asis'-----------------
data <- read.csv("a2_data_group_2.csv")
data <- data[, !(names(data) %in% c("X", "Unnamed..0"))]
data_num <- data[sapply(data, is.numeric)]

cat("\\begin{center}")
stargazer(
  data_num,
  type = "latex",
  title = "Summary Statistics of Weather Variables (Paris, 2010–2021)",
  digits = 2,
  summary.stat = c("min", "p25", "median", "mean", "p75", "max", "sd"),
  font.size = "small",
  label = "tab:summary_weather",
  float.env = "table",
  header = FALSE,
  table.placement = "H"
)
cat("\\end{center}")


## ----echo=FALSE, warning=FALSE, message=FALSE, results='asis'-----------------
correlations_df <- data.frame(
  Variable = c(
    "Tmax", "perceived_max", "mean_temp", "perceived_mean", "perceived_min", "tmin",
    "reference_evapotranspiration", "daylight_duration", "shortwave_radiation",
    "sunshine_duration", "dominant_wind_direction", "rain_sum", "precipitation_sum",
    "max_wind_gusts", "snowfall_sum", "precipitation_hours", "max_wind_speed"
  ),
  Correlation = c(
    0.93, 0.93, 0.92, 0.92, 0.86, 0.84, 0.82, 0.76, 0.72, 0.58,
    -0.01, -0.07, -0.09, -0.17, -0.17, -0.18, -0.23
  )
)
stargazer(
  correlations_df,
  summary = FALSE,
  title = "Correlation of Independent Variables with Next-day Maximum Temperature (Tmax)",
  label = "tab:correlation_tmax",
  digits = 2,
  font.size = "small",
  float.env = "table",
  type = "latex",
  header = FALSE,
  table.placement = "H",
  align = FALSE
)


## ----echo=FALSE, warning=FALSE, message=FALSE, results='asis'-----------------
ggplot(df, aes(x = Date, y = y)) +
  geom_line(color = "steelblue") +
  geom_vline(xintercept = max(train$Date),
             linetype = "dashed", color = "red") +
  labs(x = "Date", y = "Next-day Tmax") +
  theme_minimal()


## ----echo=FALSE---------------------------------------------------------------
screeplot(res, main = "Scree Plot - PCA on Training Data")


## ----echo=FALSE, message=FALSE, warning=FALSE---------------------------------
invisible(capture.output(
  paran(X_train, graph = TRUE)
))


## ----echo=FALSE---------------------------------------------------------------
fviz_pca_biplot(res,
                geom.ind = "point", pointshape = 16, pointsize = 0.7, alpha.ind = 0.15,
                col.ind = "grey70", col.var = "firebrick", repel = TRUE,
                select.var = list(contrib = 12)  # label top 12 vars
)



## ----echo=FALSE, message=FALSE, warning=FALSE, results='asis'-----------------
# Prepare the PCA loadings as a proper data frame
loadings_df <- as.data.frame(round(res$loadings[, 1:4], 2))
loadings_mat <- as.matrix(loadings_df)
rownames(loadings_mat) <- rownames(loadings_df)

# Output a clean LaTeX table
stargazer(
  loadings_mat,
  summary = FALSE,
  title = "Principal component loadings for the first four components.",  
  digits = 2,
  label = "tab:loadings",
  font.size = "small",
  header = FALSE, 
  float.env = "table",
  type = "latex"
)


## ----echo=FALSE, message=FALSE, warning=FALSE, results='asis'-----------------
# Prepare top 10 variables best explained by PC1
explained_by_pc1_df <- data.frame(
  Variable = names(head(explained_by_pc1, 10)),
  Explained_Variance = round(head(explained_by_pc1, 10), 3),
  stringsAsFactors = FALSE
)
rownames(explained_by_pc1_df) <- NULL

# Generate clean LaTeX table
stargazer(
  explained_by_pc1_df,
  summary = FALSE,
  title = "Top 10 Variables Best Explained by the First Principal Component (PC1)",              
  label = "tab:pc1_explained",
  digits = 3,
  font.size = "small",
  float.env = "table",
  type = "latex",
  header = FALSE,                    
  table.placement = "!h",
  align = FALSE                      
)



## ----echo=FALSE, message=FALSE, warning=FALSE, results='asis'-----------------

# Round results for readability
pcr_sensitivity_df <- data.frame(
  Components = pcr_sensitivity$Components,
  RMSE = round(pcr_sensitivity$RMSE, 3),
  MAE = round(pcr_sensitivity$MAE, 3),
  R2 = round(pcr_sensitivity$R2, 3),
  stringsAsFactors = FALSE
)

# Ensure no row names
rownames(pcr_sensitivity_df) <- NULL

# Generate clean LaTeX table
stargazer(
  pcr_sensitivity_df,
  summary = FALSE,
  title = "Sensitivity analysis of PCR model performance for different numbers of principal components.",
  label = "tab:pcr_sensitivity",
  digits = 3,
  font.size = "small",
  float.env = "table",
  type = "latex",
  header = FALSE,
  table.placement = "!h",
  align = FALSE
)




## ----echo=FALSE, results='asis'-----------------------------------------------
# Extract all R code chunks from the current Rmd file
code_file <- knitr::purl("Assignment2- Group2-2.Rmd", quiet = TRUE)

# Print a header for the appendix
cat("## Appendix 2. All R Code\n\n")

# Read and display the extracted R code
cat("```r\n")
cat(readLines(code_file), sep = "\n")
cat("\n```")

