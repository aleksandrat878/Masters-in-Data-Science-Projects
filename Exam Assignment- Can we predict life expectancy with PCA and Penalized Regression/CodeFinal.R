if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, corrplot, psych, factoextra, car, boot,pls, glmnet, caret, patchwork, paran, ggplot2, stargazer, permute)

life_expectancy <- read_csv("life_expectancy_data-1.csv")
predictions <- read_csv("predictions-1.csv")
pred_countries <- read_csv("648925at_health_nutrition_population.csv")

#---------- Descriptive Analysis and Data Prepration ------
data_full <- pred_countries %>% inner_join(life_expectancy, by = "Country")

data_full <- data_full %>%
  dplyr::rename(
    fertility_rate          = `Fertility rate, total (births per woman)`,
    health_exp_pc           = `Current health expenditure per capita (current US$)`,
    diabetes_rate           = `Diabetes prevalence (% of population ages 20 to 79)`,
    gov_health_exp_pc       = `Domestic general government health expenditure per capita (current US$)`,
    priv_health_exp_pc      = `Domestic private health expenditure per capita (current US$)`,
    gni_pc                  = `GNI per capita, Atlas method (current US$)`,
    immun_dpt               = `Immunization, DPT (% of children ages 12-23 months)`,
    immun_hepb3             = `Immunization, HepB3 (% of one-year-old children)`,
    immun_hib3              = `Immunization, Hib3 (% of children ages 12-23 months)`,
    immun_measles           = `Immunization, measles (% of children ages 12-23 months)`,
    immun_pol3              = `Immunization, Pol3 (% of one-year-old children)`,
    tb_incidence            = `Incidence of tuberculosis (per 100,000 people)`,
    labor_force_total       = `Labor force, total`,
    net_migration           = `Net migration`,
    oop_exp_pc              = `Out-of-pocket expenditure per capita (current US$)`,
    access_water_basic      = `People using at least basic drinking water services (% of population)`,
    access_sanitation       = `People using at least basic sanitation services (% of population)`,
    pop_growth              = `Population growth (annual %)`,
    population_total        = `Population, total`,
    overweight_rate         = `Prevalence of overweight (% of adults)`,
    hypertension_rate       = `Prevalence of hypertension (% of adults ages 30-79)`,
    rural_pop               = `Rural population`,
    sex_ratio_birth         = `Sex ratio at birth (male births per female births)`,
    alcohol_pc              = `Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)`,
    tb_death_rate           = `Tuberculosis death rate (per 100,000 people)`,
    tb_treatment_success    = `Tuberculosis treatment success rate (% of new cases)`,
    unemployment_rate       = `Unemployment, total (% of total labor force)`,
    urban_pop_growth        = `Urban population growth (annual %)`,
    urban_pop_total         = `Urban population`,
    id_column               = ...1,
    life_exp                = `Life expectancy at birth, total (years)`) %>%
  dplyr::select(-Country, -id_column)
colSums(is.na(data_full))

# Correlation
cor_matrix <- cor(data_full %>% dplyr::select(where(is.numeric)), use = "pairwise.complete.obs")
strong_vars <- unique(rownames(cor_matrix)[apply(abs(cor_matrix) > 0.8 & abs(cor_matrix) < 1, 1, any)])
cor_strong <- cor_matrix[strong_vars, strong_vars]
png("correlation_plot_strong.png", width = 2000, height = 1600, res = 300)
corrplot::corrplot(cor_strong, method = "color", type = "upper",
                   tl.col = "black", tl.cex = 0.9, number.cex = 0.7,
                   col = colorRampPalette(c("red", "white", "blue"))(200),
                   mar = c(0, 0, 2, 0), title = "Variables with |Correlation| > 0.7")
dev.off()

cor_top <- data_full %>%
  dplyr::select(life_exp, fertility_rate, health_exp_pc, oop_exp_pc, gni_pc, access_water_basic, access_sanitation, immun_dpt,
                tb_incidence, tb_death_rate) %>% dplyr::select(where(is.numeric))   
png("correlation_plot.png", width = 2000, height = 1600, res = 300)
corrplot::corrplot(cor(cor_top, use = "pairwise.complete.obs"), method = "color", type = "upper",addCoef.col = "black",   tl.col = "black", tl.cex = 0.9, number.cex = 0.7,
                   col = colorRampPalette(c("red", "white", "blue"))(200))
dev.off()

# Check the distibution
p_hist <- data_full %>% pivot_longer(-life_exp, names_to = "Variable", values_to = "Value") %>% ggplot(aes(x = Value)) + 
  geom_histogram(bins = 25, fill = "skyblue", color = "white") + 
  facet_wrap(~Variable, scales = "free") + 
  theme_minimal()

ggsave("variable_distributions.png", plot = p_hist, width = 12, height = 10, dpi = 300)
# Check outliers
p_box <- data_full %>%
  pivot_longer(-life_exp, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = "", y = Value)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red") +
  facet_wrap(~Variable, scales = "free") +
  theme_bw() +
  labs(title = "Boxplots for Outlier Detection")
ggsave("outlier_boxplots.png", plot = p_box,
       width = 12, height = 10, dpi = 300)

# ---------------- Helper Function: Preprocessing & Train/Test Split ----------------
to_log <- c("gni_pc", "gov_health_exp_pc", "health_exp_pc", "priv_health_exp_pc", "oop_exp_pc", "population_total", "rural_pop", "urban_pop_total", "labor_force_total", "net_migration", "tb_incidence", "tb_death_rate", "access_sanitation", "access_water_basic", "diabetes_rate", "fertility_rate")
set.seed(123)
idx_global <- createDataPartition(data_full$life_exp, p = 0.8, list = FALSE)
train_global <- data_full[idx_global, ]
test_global  <- data_full[-idx_global, ]
preprocess_data <- function(train_raw, test_raw, to_log, target = "life_exp") {
  train_medians <- train_raw %>% summarise(across(where(is.numeric), ~ median(.x, na.rm = TRUE))) # Impute medians on training only
  impute_fun <- function(df) mutate(df, across(where(is.numeric), ~ ifelse(is.na(.x), train_medians[[cur_column()]], .x)))
  train <- impute_fun(train_raw)
  test  <- impute_fun(test_raw)
  # Log transform (training reference)
  log_fun <- function(df, ref) mutate(
    df,
    across(all_of(to_log),
           ~ log1p(.x - min(ref[[cur_column()]], na.rm = TRUE) + 1)))
  train <- log_fun(train, train)
  test  <- log_fun(test, train)
  # Scale predictors only
  pred_cols <- setdiff(names(train), target)
  scale_fun <- function(df, ref) mutate(
    df,
    across(all_of(pred_cols),
           ~ (.x - mean(ref[[cur_column()]], na.rm = TRUE)) /
             sd(ref[[cur_column()]], na.rm = TRUE)))
  train_scaled <- scale_fun(train, train)
  test_scaled  <- scale_fun(test, train)
  # Reattach target (unscaled)
  train_scaled[[target]] <- train[[target]]
  test_scaled[[target]]  <- test[[target]]
  list(train = train_scaled, test = test_scaled)
}
splits_all <- preprocess_data(train_global, test_global, to_log)
train_all <- splits_all$train
test_all  <- splits_all$test

# ------------------- BASELINE: OLS WITH INTERACTIONS -------------------
ols_full <- lm(life_exp ~ ., data = train_all)
ols_step <- stepAIC(ols_full, direction = "both", trace = FALSE)
summary(ols_step)
vif(ols_step)

# To better capture complex relationships, the model included a squared term for fertility, allowing for nonlinear effects, as well as two interaction terms: (1) between fertility and GNI per capita—reflecting that the longevity benefits of income may diminish in high-fertility settings—and (2) between health and private spending, to capture inefficiencies when healthcare systems rely heavily on out-of-pocket costs. 

# Evaluate predictive performance
step_pred  <- predict(ols_step, newdata = test_all)
step_resid <- test_all$life_exp - step_pred

# --- OLS Diagnostics ---
png("ols_residual_diagnostics.png", width = 2000, height = 1000, res = 300)
par(mfrow = c(1,2))
qqnorm(step_resid); qqline(step_resid, col="red")
plot(step_pred, step_resid, main="Residuals vs Fitted", xlab="Fitted", ylab="Residuals")
abline(h=0, col="red")
par(mfrow = c(1,1))
dev.off()

step_metrics <- list(
  rmse = sqrt(mean(step_resid^2)),
  mae  = mean(abs(step_resid)),
  r2   = 1 - sum(step_resid^2) /
    sum((test_all$life_exp - mean(test_all$life_exp))^2)
)
print(step_metrics)

set.seed(123)
cv_ols_step <- train(life_exp ~ ., data = train_all[, c(all.vars(formula(ols_step)))],
                     method = "lm", trControl = trainControl(method = "cv", number = 10))
cv_ols_step$results






ols_model <- lm(life_exp ~ fertility_rate + I(fertility_rate^2) +
                  health_exp_pc + oop_exp_pc + gni_pc +
                  access_water_basic + access_sanitation + immun_dpt +
                  tb_incidence + tb_death_rate +
                  fertility_rate:gni_pc + health_exp_pc:oop_exp_pc,  data = train_all)


ols_pred  <- predict(ols_model, newdata = test_all)
ols_resid <- test_all$life_exp - ols_pred

# --- OLS Diagnostics ---
png("ols_residual_diagnostics.png", width = 2000, height = 1000, res = 300)
par(mfrow = c(1,2))
qqnorm(ols_resid); qqline(ols_resid, col="red")
plot(ols_pred, ols_resid, main="Residuals vs Fitted", xlab="Fitted", ylab="Residuals")
abline(h=0, col="red")
par(mfrow = c(1,1))
dev.off()

ols_metrics <- list(
  rmse = sqrt(mean(ols_resid^2)),
  mae  = mean(abs(ols_resid)),
  r2   = 1 - sum(ols_resid^2) /
    sum((test_all$life_exp - mean(test_all$life_exp))^2)
)
print(ols_metrics)

# ------------------- PCA & PRINCIPAL COMPONENT REGRESSION ------------------
X_train <- train_all %>% dplyr::select(-life_exp)
y_train <- train_all$life_exp
X_test  <- test_all %>% dplyr::select(-life_exp)
y_test  <- test_all$life_exp

X_train <- X_train[, sapply(X_train, function(x) sd(x, na.rm = TRUE) > 0)]
X_test  <- X_test[, names(X_train)]
X_train[is.na(X_train)] <- 0
X_test[is.na(X_test)] <- 0

res_pca <- princomp(X_train, cor = TRUE, scores = TRUE)

p_scree <- fviz_eig(res_pca, addlabels = TRUE, main = "Scree Plot - PCA (Training Data)")
ggsave("pca_scree_plot_corrected.png", plot = p_scree, width = 8, height = 6, dpi = 300)

# --- 1. Permutation Test (compare observed vs random eigenvalues) ---
permtestPCA <- function(data, nperm = 1000) {
  set.seed(123)
  eigen_real <- eigen(cor(data, use = "pairwise.complete.obs"))$values
  eigen_perm <- replicate(nperm, {
    perm_data <- apply(data, 2, sample)
    eigen(cor(perm_data, use = "pairwise.complete.obs"))$values
  })
  ci_lower <- apply(eigen_perm, 1, quantile, 0.025)
  ci_upper <- apply(eigen_perm, 1, quantile, 0.975)
  
  df_perm <- data.frame(
    Component = 1:length(eigen_real),
    Observed = eigen_real,
    Lower = ci_lower,
    Upper = ci_upper)
  
  ggplot(df_perm, aes(x = Component)) +
    geom_line(aes(y = Observed, color = "Observed"), linewidth = 1) +
    geom_point(aes(y = Observed, color = "Observed"), size = 2) +
    geom_line(aes(y = Lower, color = "2.5% CI"), linetype = "dashed") +
    geom_line(aes(y = Upper, color = "97.5% CI"), linetype = "dashed") +
    scale_color_manual(values = c("Observed" = "red", "2.5% CI" = "blue", "97.5% CI" = "blue")) +
    labs(title = "Permutation Test PCA", x = "Component", y = "Eigenvalue", color = "") +
    theme_minimal(base_size = 13)
}

set.seed(123)
p_perm <- permtestPCA(X_train)
ggsave("pca_permutation_test.png", plot = p_perm, width = 8, height = 6, dpi = 300)

# --- 2. Bootstrap Test for Eigenvalues (stability + Kaiser rule) ---
my_boot_pca <- function(x, ind) {
  res <- princomp(x[ind, ], cor = TRUE)
  res$sdev^2
}
set.seed(123)
fit.boot <- boot(data = X_train, statistic = my_boot_pca, R = 1000)
eigs.boot <- fit.boot$t
obs_eigs <- res_pca$sdev^2

# Plot bootstrap distributions
png("pca_bootstrap_boxplot.png", width = 2000, height = 1600, res = 300)
boxplot(eigs.boot, col = "beige", las = 1,
        main = "Bootstrap Distribution of Eigenvalues",
        xlab = "Principal Component", ylab = "Eigenvalue")
points(colMeans(eigs.boot), pch = 19, col = "darkblue")
abline(h = 1, col = "red", lty = 2)
dev.off()

# --- Histogram of Bootstrap Distribution for the First Eigenvalue ---
png("pca_bootstrap_histogram_pc1.png", width = 2000, height = 1600, res = 300)
hist(eigs.boot[, 1],
     xlab = "Eigenvalue 1",
     main = "Bootstrap Distribution of Eigenvalue 1",
     col = "lightblue", border = "white",
     las = 1, breaks = 25)
perc.alpha <- quantile(eigs.boot[, 1], c(0.025, 0.975))
abline(v = perc.alpha, col = "green", lwd = 2)
abline(v = obs_eigs[1], col = "red", lwd = 2)
legend("topright", legend = c("Observed", "95% CI limits"),
       col = c("red", "green"), lty = 1, lwd = 2, bty = "n")
dev.off()

# Compute 95% CI for each eigenvalue
ci_eigs <- apply(eigs.boot, 2, quantile, c(0.025, 0.975))
ci_table <- data.frame(
  Component = 1:length(obs_eigs),
  Observed = round(obs_eigs, 3),
  CI_Lower = round(ci_eigs[1, ], 3),
  CI_Upper = round(ci_eigs[2, ], 3)
)
print(ci_table)

# --- 3. Bootstrap Test for total variance ≥ 70% (VAF test) ---
boot_vaf <- apply(eigs.boot, 1, function(x) sum(x[1:5]) / sum(x))
ci_vaf <- quantile(boot_vaf, c(0.025, 0.975))
cat("95% Bootstrap CI for cumulative variance (5 PCs):", round(ci_vaf, 3), "\n")

# --- 4. Decide number of components to retain ---
n_components <- sum(ci_eigs[1, ] > 1)   

n_components <- 5
# Project data onto the first 5 PCs
pc_train <- predict(res_pca, newdata = X_train)
pc_test  <- predict(res_pca, newdata = X_test)

train_scores <- as.data.frame(pc_train[, 1:n_components, drop = FALSE])
test_scores  <- as.data.frame(pc_test[, 1:n_components, drop = FALSE])
train_scores$life_exp <- y_train
test_scores$life_exp  <- y_test

# Fit regression on principal components
library(pls)

# ------------------- PCA & PRINCIPAL COMPONENT REGRESSION (using pls::pcr) -------------------
X_train <- train_all %>% dplyr::select(-life_exp)
y_train <- train_all$life_exp
X_test  <- test_all %>% dplyr::select(-life_exp)
y_test  <- test_all$life_exp

X_train <- X_train[, sapply(X_train, function(x) sd(x, na.rm = TRUE) > 0)]
X_test  <- X_test[, names(X_train)]
X_train[is.na(X_train)] <- 0
X_test[is.na(X_test)] <- 0

res_pca <- princomp(X_train, cor = TRUE, scores = TRUE)

p_scree <- fviz_eig(res_pca, addlabels = TRUE, main = "Scree Plot - PCA (Training Data)")
ggsave("pca_scree_plot_corrected.png", plot = p_scree, width = 8, height = 6, dpi = 300)

# --- 1. Permutation Test (compare observed vs random eigenvalues) ---
permtestPCA <- function(data, nperm = 1000) {
  set.seed(123)
  eigen_real <- eigen(cor(data, use = "pairwise.complete.obs"))$values
  eigen_perm <- replicate(nperm, {
    perm_data <- apply(data, 2, sample)
    eigen(cor(perm_data, use = "pairwise.complete.obs"))$values
  })
  ci_lower <- apply(eigen_perm, 1, quantile, 0.025)
  ci_upper <- apply(eigen_perm, 1, quantile, 0.975)
  
  df_perm <- data.frame(
    Component = 1:length(eigen_real),
    Observed = eigen_real,
    Lower = ci_lower,
    Upper = ci_upper)
  
  ggplot(df_perm, aes(x = Component)) +
    geom_line(aes(y = Observed, color = "Observed"), linewidth = 1) +
    geom_point(aes(y = Observed, color = "Observed"), size = 2) +
    geom_line(aes(y = Lower, color = "2.5% CI"), linetype = "dashed") +
    geom_line(aes(y = Upper, color = "97.5% CI"), linetype = "dashed") +
    scale_color_manual(values = c("Observed" = "red", "2.5% CI" = "blue", "97.5% CI" = "blue")) +
    labs(title = "Permutation Test PCA", x = "Component", y = "Eigenvalue", color = "") +
    theme_minimal(base_size = 13)}
set.seed(123)
p_perm <- permtestPCA(X_train)
ggsave("pca_permutation_test.png", plot = p_perm, width = 8, height = 6, dpi = 300)

# --- 2. Bootstrap Test for Eigenvalues (stability + Kaiser rule) ---
my_boot_pca <- function(x, ind) {
  res <- princomp(x[ind, ], cor = TRUE)
  res$sdev^2
}
set.seed(123)
fit.boot <- boot(data = X_train, statistic = my_boot_pca, R = 1000)
eigs.boot <- fit.boot$t
obs_eigs <- res_pca$sdev^2
# Plot bootstrap distributions
png("pca_bootstrap_boxplot.png", width = 2000, height = 1600, res = 300)
boxplot(eigs.boot, col = "beige", las = 1,
        main = "Bootstrap Distribution of Eigenvalues",
        xlab = "Principal Component", ylab = "Eigenvalue")
points(colMeans(eigs.boot), pch = 19, col = "darkblue")
abline(h = 1, col = "red", lty = 2)
dev.off()
# --- Histogram of Bootstrap Distribution for the First Eigenvalue ---
png("pca_bootstrap_histogram_pc1.png", width = 2000, height = 1600, res = 300)
hist(eigs.boot[, 1],
     xlab = "Eigenvalue 1",
     main = "Bootstrap Distribution of Eigenvalue 1",
     col = "lightblue", border = "white",
     las = 1, breaks = 25)
perc.alpha <- quantile(eigs.boot[, 1], c(0.025, 0.975))
abline(v = perc.alpha, col = "green", lwd = 2)
abline(v = obs_eigs[1], col = "red", lwd = 2)
legend("topright", legend = c("Observed", "95% CI limits"),
       col = c("red", "green"), lty = 1, lwd = 2, bty = "n")
dev.off()

# Compute 95% CI for each eigenvalue
ci_eigs <- apply(eigs.boot, 2, quantile, c(0.025, 0.975))
ci_table <- data.frame(
  Component = 1:length(obs_eigs),
  Observed = round(obs_eigs, 3),
  CI_Lower = round(ci_eigs[1, ], 3),
  CI_Upper = round(ci_eigs[2, ], 3))
print(ci_table)

# --- 3. Bootstrap Test for total variance ≥ 70% (VAF test) ---
boot_vaf <- apply(eigs.boot, 1, function(x) sum(x[1:5]) / sum(x))
ci_vaf <- quantile(boot_vaf, c(0.025, 0.975))
cat("95% Bootstrap CI for cumulative variance (5 PCs):", round(ci_vaf, 3), "\n")

# --- 4. Decide number of components to retain ---
n_components <- sum(ci_eigs[1, ] > 1)   
n_components <- 5



# --- 3. Variable contribution plots -------------------------------------------------------
p_pca_var <- fviz_pca_var(
  res_pca,
  col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE,
  title = "PCA - Variable Contributions"
)
ggsave("pca_variable_contributions.png", plot = p_pca_var,
       width = 8, height = 6, dpi = 300)

# Top-10 variables by absolute loading per PC
loadings_df <- as.data.frame(res_pca$loadings[, 1:n_components]) %>%
  rownames_to_column("Variable") %>%
  pivot_longer(-Variable, names_to = "Component", values_to = "Loading") %>%
  mutate(AbsLoading = abs(Loading)) %>%
  group_by(Component) %>%
  slice_max(order_by = AbsLoading, n = 10) %>%
  ungroup()

p_top_vars <- ggplot(loadings_df, aes(x = reorder(Variable, AbsLoading),
                                      y = AbsLoading, fill = Component)) +
  geom_col(show.legend = TRUE) +
  coord_flip() +
  facet_wrap(~Component, scales = "free_y") +
  labs(title = "Top 10 Most Influential Variables per Principal Component",
       x = "Variable", y = "Absolute Loading") +
  theme_minimal(base_size = 12)
ggsave("pca_top_variable_contributions.png",
       plot = p_top_vars, width = 10, height = 8, dpi = 300)










# ------------
# Variable contribution visualization
p_pca_var <- fviz_pca_var(
  res_pca,
  col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE,
  title = "PCA - Variable Contributions"
)
ggsave("pca_variable_contributions.png", plot = p_pca_var, width = 8, height = 6, dpi = 300)

# Top contributing variables per component (PC1–PC5)
loadings_df <- as.data.frame(res_pca$loadings[, 1:n_components]) %>%
  rownames_to_column("Variable") %>%
  pivot_longer(-Variable, names_to = "Component", values_to = "Loading") %>%
  mutate(AbsLoading = abs(Loading)) %>%
  group_by(Component) %>%
  slice_max(order_by = AbsLoading, n = 10) %>%
  ungroup()

p_top_vars <- ggplot(loadings_df, aes(x = reorder(Variable, AbsLoading), 
                                      y = AbsLoading, fill = Component)) +
  geom_col(show.legend = TRUE) +
  coord_flip() +
  facet_wrap(~Component, scales = "free_y") +
  labs(title = "Top 10 Most Influential Variables per Principal Component",
       x = "Variable", y = "Absolute Loading") +
  theme_minimal(base_size = 12)
ggsave("pca_top_variable_contributions.png", plot = p_top_vars, width = 10, height = 8, dpi = 300)

# ---------------- Elastic Net ----------------------
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
alpha_grid <- seq(0, 1, length = 10)
lambda_seq <- 10^seq(2, -3, length = 100)

set.seed(123)
elasticNet <- train(
  life_exp ~ ., data = train_all,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = alpha_grid, lambda = lambda_seq),
  trControl = ctrl, standardize = TRUE
)

best <- elasticNet$bestTune
final_en <- glmnet(
  x = as.matrix(dplyr::select(train_all, -life_exp)),
  y = train_all$life_exp,
  alpha = best$alpha, lambda = best$lambda, standardize = TRUE
)
pred_en <- as.numeric(predict(final_en, newx = as.matrix(dplyr::select(test_all, -life_exp))))
enet_resid <- test_all$life_exp - pred_en  
print(best$alpha)
print(best$lambda)

en_metrics <- list(
  rmse = sqrt(mean(enet_resid^2)),
  mae  = mean(abs(enet_resid)),
  r2   = 1 - sum(enet_resid^2) / sum((test_all$life_exp - mean(test_all$life_exp))^2)
)
print(en_metrics)

# View important variables
enet_coef <- coef(final_en)
coef_df <- data.frame(
  Variable = rownames(enet_coef),
  Coefficient = as.numeric(enet_coef)
) %>%
  filter(Coefficient != 0 & Variable != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))
# Plot coefficients
p_coef <- ggplot(coef_df, aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Elastic Net Non-Zero Coefficients",
       y = "Coefficient", x = "")
ggsave("elasticnet_coefficients.png", plot = p_coef, width = 10, height = 8, dpi = 300)

# ------------------- MODEL PERFORMANCE COMPARISON -------------------
comparison <- tibble(
  Model = c("OLS (Interaction)", "PCR (5 PCs)", "Elastic Net"),
  RMSE  = c(ols_metrics$rmse, pcr_rmse, en_metrics$rmse),
  MAE   = c(ols_metrics$mae,  pcr_mae,  en_metrics$mae),
  R2    = c(ols_metrics$r2,   pcr_r2,   en_metrics$r2)
) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))
# Print comparison table
print(comparison)
# ------------------- RESIDUAL DIAGNOSTICS -------------------
plot_residuals <- function(fitted, resid, title, color) {
  df <- data.frame(Fitted = fitted, Residuals = resid)
  p1 <- ggplot(df, aes(Fitted, Residuals)) +
    geom_point(alpha = 0.6, color = color) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = paste(title, "Residuals vs Fitted")) +
    theme_minimal()
  p2 <- ggplot(df, aes(Residuals)) +
    geom_histogram(bins = 25, fill = color, color = "white") +
    labs(title = paste(title, "Residual Distribution")) +
    theme_minimal() 
  p1 / p2}
# Generate plots for all models in one go
resid_plots <- list(
  OLS = plot_residuals(ols_pred, test_all$life_exp - ols_pred, "OLS", "steelblue"),
  PCR = plot_residuals(fitted(pcr_model), residuals(pcr_model), "PCR", "orange"),
  EN  = plot_residuals(as.numeric(pred_en), 
                       as.numeric(test_all$life_exp - pred_en),
                       "Elastic Net", "firebrick"))
# Combine all models side by side
final_resid_plot <- resid_plots$OLS | resid_plots$PCR | resid_plots$EN +
  plot_annotation(title = "Residual Diagnostics: OLS vs PCR vs Elastic Net",
                  theme = theme(plot.title = element_text(size = 16, face = "bold")))
ggsave("residual_diagnostics.png", plot = final_resid_plot,
       width = 14, height = 6, dpi = 300)






# ---------------- Helper Function: Preprocessing & Train/Test Split ----------------
to_log <- c("gni_pc", "gov_health_exp_pc", "health_exp_pc", "priv_health_exp_pc", "oop_exp_pc", "population_total", "rural_pop", "urban_pop_total", "labor_force_total", "net_migration", "tb_incidence", "tb_death_rate", "access_sanitation", "access_water_basic", "diabetes_rate", "fertility_rate")
set.seed(123)
idx_global <- createDataPartition(data_full$life_exp, p = 0.8, list = FALSE)
train_global <- data_full[idx_global, ]
test_global  <- data_full[-idx_global, ]
preprocess_data <- function(train_raw, test_raw, to_log, target = "life_exp") {
  train_medians <- train_raw %>% summarise(across(where(is.numeric), ~ median(.x, na.rm = TRUE))) # Impute medians on training only
  impute_fun <- function(df) mutate(df, across(where(is.numeric), ~ ifelse(is.na(.x), train_medians[[cur_column()]], .x)))
  train <- impute_fun(train_raw)
  test  <- impute_fun(test_raw)
  # Log transform (training reference)
  log_fun <- function(df, ref) mutate(
    df,
    across(all_of(to_log),
           ~ log1p(.x - min(ref[[cur_column()]], na.rm = TRUE) + 1)))
  train <- log_fun(train, train)
  test  <- log_fun(test, train)
  # Scale predictors only
  pred_cols <- setdiff(names(train), target)
  scale_fun <- function(df, ref) mutate(
    df,
    across(all_of(pred_cols),
           ~ (.x - mean(ref[[cur_column()]], na.rm = TRUE)) /
             sd(ref[[cur_column()]], na.rm = TRUE)))
  train_scaled <- scale_fun(train, train)
  test_scaled  <- scale_fun(test, train)
  # Reattach target (unscaled)
  train_scaled[[target]] <- train[[target]]
  test_scaled[[target]]  <- test[[target]]
  list(train = train_scaled, test = test_scaled)
}
splits_all <- preprocess_data(train_global, test_global, to_log)
train_all <- splits_all$train
test_all  <- splits_all$test
# ------------------- BASELINE: OLS WITH INTERACTIONS -------------------
ols_full <- lm(life_exp ~ ., data = train_all)
ols_step <- stepAIC(ols_full, direction = "both", trace = FALSE)
step_pred  <- predict(ols_step, newdata = test_all)
step_resid <- test_all$life_exp - step_pred
par(mfrow = c(1,2)) # --- OLS Diagnostics ---
qqnorm(step_resid); qqline(step_resid, col="red")
plot(step_pred, step_resid, main="Residuals vs Fitted", xlab="Fitted", ylab="Residuals")
abline(h=0, col="red")
par(mfrow = c(1,1))
step_metrics <- list(
  rmse = sqrt(mean(step_resid^2)),
  mae  = mean(abs(step_resid)),
  r2   = 1 - sum(step_resid^2) /
    sum((test_all$life_exp - mean(test_all$life_exp))^2))
set.seed(123)
cv_ols_step <- train(life_exp ~ ., data = train_all[, c(all.vars(formula(ols_step)))],
                     method = "lm", trControl = trainControl(method = "cv", number = 10))
# ------------------- PCA & PRINCIPAL COMPONENT REGRESSION ------------------
X_train <- train_all %>% dplyr::select(-life_exp)
y_train <- train_all$life_exp
X_test  <- test_all %>% dplyr::select(-life_exp)
y_test  <- test_all$life_exp
X_train <- X_train[, sapply(X_train, function(x) sd(x, na.rm = TRUE) > 0)]
X_test  <- X_test[, names(X_train)]
X_train[is.na(X_train)] <- 0
X_test[is.na(X_test)] <- 0
res_pca <- princomp(X_train, cor = TRUE, scores = TRUE)
p_scree <- fviz_eig(res_pca, addlabels = TRUE, main = "Scree Plot - PCA (Training Data)")
ggsave("pca_scree_plot_corrected.png", plot = p_scree, width = 8, height = 6, dpi = 300)
# --- 1. Permutation Test (compare observed vs random eigenvalues) ---
permtestPCA <- function(data, nperm = 1000) {
  set.seed(123)
  eigen_real <- eigen(cor(data, use = "pairwise.complete.obs"))$values
  eigen_perm <- replicate(nperm, {
    perm_data <- apply(data, 2, sample)
    eigen(cor(perm_data, use = "pairwise.complete.obs"))$values})
  ci_lower <- apply(eigen_perm, 1, quantile, 0.025)
  ci_upper <- apply(eigen_perm, 1, quantile, 0.975)
  df_perm <- data.frame(
    Component = 1:length(eigen_real),
    Observed = eigen_real,
    Lower = ci_lower,
    Upper = ci_upper)
  ggplot(df_perm, aes(x = Component)) +
    geom_line(aes(y = Observed, color = "Observed"), linewidth = 1) +
    geom_point(aes(y = Observed, color = "Observed"), size = 2) +
    geom_line(aes(y = Lower, color = "2.5% CI"), linetype = "dashed") +
    geom_line(aes(y = Upper, color = "97.5% CI"), linetype = "dashed") +
    scale_color_manual(values = c("Observed" = "red", "2.5% CI" = "blue", "97.5% CI" = "blue")) +
    labs(title = "Permutation Test PCA", x = "Component", y = "Eigenvalue", color = "") +
    theme_minimal(base_size = 13)}
# --- 2. Bootstrap Test for Eigenvalues (stability + Kaiser rule) ---
my_boot_pca <- function(x, ind) {
  res <- princomp(x[ind, ], cor = TRUE)
  res$sdev^2}
set.seed(123)
fit.boot <- boot(data = X_train, statistic = my_boot_pca, R = 1000)
eigs.boot <- fit.boot$t
obs_eigs <- res_pca$sdev^2
# Plot bootstrap distributions
boxplot(eigs.boot, col = "beige", las = 1,
        main = "Bootstrap Distribution of Eigenvalues",
        xlab = "Principal Component", ylab = "Eigenvalue")
points(colMeans(eigs.boot), pch = 19, col = "darkblue")
abline(h = 1, col = "red", lty = 2)
# --- Histogram of Bootstrap Distribution for the First Eigenvalue ---
hist(eigs.boot[, 1],
     xlab = "Eigenvalue 1",
     main = "Bootstrap Distribution of Eigenvalue 1",
     col = "lightblue", border = "white",
     las = 1, breaks = 25)
perc.alpha <- quantile(eigs.boot[, 1], c(0.025, 0.975))
abline(v = perc.alpha, col = "green", lwd = 2)
abline(v = obs_eigs[1], col = "red", lwd = 2)
legend("topright", legend = c("Observed", "95% CI limits"),
       col = c("red", "green"), lty = 1, lwd = 2, bty = "n")
# Compute 95% CI for each eigenvalue
ci_eigs <- apply(eigs.boot, 2, quantile, c(0.025, 0.975))
ci_table <- data.frame(
  Component = 1:length(obs_eigs),
  Observed = round(obs_eigs, 3),
  CI_Lower = round(ci_eigs[1, ], 3),
  CI_Upper = round(ci_eigs[2, ], 3))
# --- 3. Bootstrap Test for total variance ≥ 70% (VAF test) ---
boot_vaf <- apply(eigs.boot, 1, function(x) sum(x[1:5]) / sum(x))
ci_vaf <- quantile(boot_vaf, c(0.025, 0.975))
# --- 4. Decide number of components to retain ---
n_components <- sum(ci_eigs[1, ] > 1)   
n_components <- 5
# Project data onto the first 5 PCs
pc_train <- predict(res_pca, newdata = X_train)
pc_test  <- predict(res_pca, newdata = X_test)
train_scores <- as.data.frame(pc_train[, 1:n_components, drop = FALSE])
test_scores  <- as.data.frame(pc_test[, 1:n_components, drop = FALSE])
train_scores$life_exp <- y_train
test_scores$life_exp  <- y_test
# Fit regression on principal components
pcr_model <- lm(life_exp ~ ., data = train_scores)
pcr_pred <- predict(pcr_model, newdata = test_scores)
pcr_resid <- test_scores$life_exp - pcr_pred
# Performance metrics
pcr_rmse <- sqrt(mean(pcr_resid^2))
pcr_mae  <- mean(abs(pcr_resid))
pcr_r2   <- 1 - sum(pcr_resid^2) / sum((y_test - mean(y_test))^2)
# Top contributing variables per component (PC1–PC5)
loadings_df <- as.data.frame(res_pca$loadings[, 1:n_components]) %>%
  rownames_to_column("Variable") %>%
  pivot_longer(-Variable, names_to = "Component", values_to = "Loading") %>%
  mutate(AbsLoading = abs(Loading)) %>%
  group_by(Component) %>%
  slice_max(order_by = AbsLoading, n = 10) %>%
  ungroup()
p_top_vars <- ggplot(loadings_df, aes(x = reorder(Variable, AbsLoading)))
# ---------------- Elastic Net ----------------------
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
alpha_grid <- seq(0, 1, length = 10)
lambda_seq <- 10^seq(2, -3, length = 100)
set.seed(123)
elasticNet <- train(
  life_exp ~ ., data = train_all,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = alpha_grid, lambda = lambda_seq),
  trControl = ctrl, standardize = TRUE)
best <- elasticNet$bestTune
final_en <- glmnet(
  x = as.matrix(dplyr::select(train_all, -life_exp)),
  y = train_all$life_exp,
  alpha = best$alpha, lambda = best$lambda, standardize = TRUE)
pred_en <- as.numeric(predict(final_en, newx = as.matrix(dplyr::select(test_all, -life_exp))))
enet_resid <- test_all$life_exp - pred_en  
print(best$alpha)
print(best$lambda)
en_metrics <- list(
  rmse = sqrt(mean(enet_resid^2)),
  mae  = mean(abs(enet_resid)),
  r2   = 1 - sum(enet_resid^2) / sum((test_all$life_exp - mean(test_all$life_exp))^2))
# View important variables
enet_coef <- coef(final_en)
coef_df <- data.frame(
  Variable = rownames(enet_coef),
  Coefficient = as.numeric(enet_coef)) %>%
  filter(Coefficient != 0 & Variable != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))
# ------------------- MODEL PERFORMANCE COMPARISON -------------------
comparison <- tibble(
  Model = c("OLS (Interaction)", "PCR (5 PCs)", "Elastic Net"),
  RMSE  = c(ols_metrics$rmse, pcr_rmse, en_metrics$rmse),
  MAE   = c(ols_metrics$mae,  pcr_mae,  en_metrics$mae),
  R2    = c(ols_metrics$r2,   pcr_r2,   en_metrics$r2)) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))
# --------- Predicting on unseen data -------
new_data_split <- preprocess_data(train_global, predictions, to_log)
pred_final <- new_data_split$test
pred_final_aligned <- pred_final[, names(dplyr::select(train_all, -life_exp))]
final_predictions <- predict(
  final_en,
  newx = as.matrix(pred_final_aligned),
  s = best$lambda)
predicted_df <- data.frame(
  Country = predictions$Country,
  Predicted_Life_Expectancy = round(as.numeric(final_predictions), 1))



# --------

ols_model <- lm(life_exp ~ fertility_rate + I(fertility_rate^2) +
                  health_exp_pc + oop_exp_pc + gni_pc +
                  access_water_basic + access_sanitation + immun_dpt +
                  tb_incidence + tb_death_rate +
                  fertility_rate:gni_pc + health_exp_pc:oop_exp_pc,  data = train_all)

ols_pred  <- predict(ols_model, newdata = test_all)
ols_resid <- test_all$life_exp - ols_pred

ols_metrics <- list(
  rmse = sqrt(mean(ols_resid^2)),
  mae  = mean(abs(ols_resid)),
  r2   = 1 - sum(ols_resid^2) /
    sum((test_all$life_exp - mean(test_all$life_exp))^2)
)
print(ols_metrics)
# -----


# ------------------- PCA & PRINCIPAL COMPONENT REGRESSION ------------------
X_train <- train_all %>% dplyr::select(-life_exp)
y_train <- train_all$life_exp
X_test  <- test_all %>% dplyr::select(-life_exp)
y_test  <- test_all$life_exp

X_train <- X_train[, sapply(X_train, function(x) sd(x, na.rm = TRUE) > 0)]
X_test  <- X_test[, names(X_train)]
X_train[is.na(X_train)] <- 0
X_test[is.na(X_test)] <- 0

res_pca <- princomp(X_train, cor = TRUE, scores = TRUE)

p_scree <- fviz_eig(res_pca, addlabels = TRUE, main = "Scree Plot - PCA (Training Data)")
ggsave("pca_scree_plot_corrected.png", plot = p_scree, width = 8, height = 6, dpi = 300)

# --- 1. Permutation Test (compare observed vs random eigenvalues) ---
permtestPCA <- function(data, nperm = 1000) {
  set.seed(123)
  eigen_real <- eigen(cor(data, use = "pairwise.complete.obs"))$values
  eigen_perm <- replicate(nperm, {
    perm_data <- apply(data, 2, sample)
    eigen(cor(perm_data, use = "pairwise.complete.obs"))$values
  })
  ci_lower <- apply(eigen_perm, 1, quantile, 0.025)
  ci_upper <- apply(eigen_perm, 1, quantile, 0.975)
  
  df_perm <- data.frame(
    Component = 1:length(eigen_real),
    Observed = eigen_real,
    Lower = ci_lower,
    Upper = ci_upper)
  
  ggplot(df_perm, aes(x = Component)) +
    geom_line(aes(y = Observed, color = "Observed"), linewidth = 1) +
    geom_point(aes(y = Observed, color = "Observed"), size = 2) +
    geom_line(aes(y = Lower, color = "2.5% CI"), linetype = "dashed") +
    geom_line(aes(y = Upper, color = "97.5% CI"), linetype = "dashed") +
    scale_color_manual(values = c("Observed" = "red", "2.5% CI" = "blue", "97.5% CI" = "blue")) +
    labs(title = "Permutation Test PCA", x = "Component", y = "Eigenvalue", color = "") +
    theme_minimal(base_size = 13)}
set.seed(123)
p_perm <- permtestPCA(X_train)
ggsave("pca_permutation_test.png", plot = p_perm, width = 8, height = 6, dpi = 300)

# --- 2. Bootstrap Test for Eigenvalues (stability + Kaiser rule) ---
my_boot_pca <- function(x, ind) {
  res <- princomp(x[ind, ], cor = TRUE)
  res$sdev^2
}
set.seed(123)
fit.boot <- boot(data = X_train, statistic = my_boot_pca, R = 1000)
eigs.boot <- fit.boot$t
obs_eigs <- res_pca$sdev^2
# Plot bootstrap distributions
png("pca_bootstrap_boxplot.png", width = 2000, height = 1600, res = 300)
boxplot(eigs.boot, col = "beige", las = 1,
        main = "Bootstrap Distribution of Eigenvalues",
        xlab = "Principal Component", ylab = "Eigenvalue")
points(colMeans(eigs.boot), pch = 19, col = "darkblue")
abline(h = 1, col = "red", lty = 2)
dev.off()
# --- Histogram of Bootstrap Distribution for the First Eigenvalue ---
png("pca_bootstrap_histogram_pc1.png", width = 2000, height = 1600, res = 300)
hist(eigs.boot[, 1],
     xlab = "Eigenvalue 1",
     main = "Bootstrap Distribution of Eigenvalue 1",
     col = "lightblue", border = "white",
     las = 1, breaks = 25)
perc.alpha <- quantile(eigs.boot[, 1], c(0.025, 0.975))
abline(v = perc.alpha, col = "green", lwd = 2)
abline(v = obs_eigs[1], col = "red", lwd = 2)
legend("topright", legend = c("Observed", "95% CI limits"),
       col = c("red", "green"), lty = 1, lwd = 2, bty = "n")
dev.off()

# Compute 95% CI for each eigenvalue
ci_eigs <- apply(eigs.boot, 2, quantile, c(0.025, 0.975))
ci_table <- data.frame(
  Component = 1:length(obs_eigs),
  Observed = round(obs_eigs, 3),
  CI_Lower = round(ci_eigs[1, ], 3),
  CI_Upper = round(ci_eigs[2, ], 3))
print(ci_table)

# --- 3. Bootstrap Test for total variance ≥ 70% (VAF test) ---
boot_vaf <- apply(eigs.boot, 1, function(x) sum(x[1:5]) / sum(x))
ci_vaf <- quantile(boot_vaf, c(0.025, 0.975))
cat("95% Bootstrap CI for cumulative variance (5 PCs):", round(ci_vaf, 3), "\n")

# --- 4. Decide number of components to retain ---
n_components <- sum(ci_eigs[1, ] > 1)   
n_components <- 5
# Project data onto the first 5 PCs
pc_train <- predict(res_pca, newdata = X_train)
pc_test  <- predict(res_pca, newdata = X_test)
train_scores <- as.data.frame(pc_train[, 1:n_components, drop = FALSE])
test_scores  <- as.data.frame(pc_test[, 1:n_components, drop = FALSE])
train_scores$life_exp <- y_train
test_scores$life_exp  <- y_test
# Fit regression on principal components
pcr_model <- lm(life_exp ~ ., data = train_scores)
pcr_pred <- predict(pcr_model, newdata = test_scores)
pcr_resid <- test_scores$life_exp - pcr_pred
# Performance metrics
pcr_rmse <- sqrt(mean(pcr_resid^2))
pcr_mae  <- mean(abs(pcr_resid))
pcr_r2   <- 1 - sum(pcr_resid^2) / sum((y_test - mean(y_test))^2)
cat("PCR (5 PCs): RMSE =", round(pcr_rmse, 3), 
    "MAE =", round(pcr_mae, 3), 
    "R² =", round(pcr_r2, 3), "\n")
# Variable contribution visualization
p_pca_var <- fviz_pca_var(
  res_pca,
  col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE,
  title = "PCA - Variable Contributions")
ggsave("pca_variable_contributions.png", plot = p_pca_var, width = 8, height = 6, dpi = 300)


#-------------



if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, corrplot, psych, factoextra, car, boot,pls, glmnet, caret, patchwork, paran, ggplot2, stargazer, permute, MASS, knitr, kableExtra, pls)

life_expectancy <- read_csv("life_expectancy_data-1.csv")
predictions <- read_csv("predictions-1.csv")
pred_countries <- read_csv("648925at_health_nutrition_population.csv")




#---------- Descriptive Analysis and Data Prepration ------
data_full <- pred_countries %>% inner_join(life_expectancy, by = "Country")

data_full <- data_full %>%
  dplyr::rename(
    fertility_rate          = `Fertility rate, total (births per woman)`,
    health_exp_pc           = `Current health expenditure per capita (current US$)`,
    diabetes_rate           = `Diabetes prevalence (% of population ages 20 to 79)`,
    gov_health_exp_pc       = `Domestic general government health expenditure per capita (current US$)`,
    priv_health_exp_pc      = `Domestic private health expenditure per capita (current US$)`,
    gni_pc                  = `GNI per capita, Atlas method (current US$)`,
    immun_dpt               = `Immunization, DPT (% of children ages 12-23 months)`,
    immun_hepb3             = `Immunization, HepB3 (% of one-year-old children)`,
    immun_hib3              = `Immunization, Hib3 (% of children ages 12-23 months)`,
    immun_measles           = `Immunization, measles (% of children ages 12-23 months)`,
    immun_pol3              = `Immunization, Pol3 (% of one-year-old children)`,
    tb_incidence            = `Incidence of tuberculosis (per 100,000 people)`,
    labor_force_total       = `Labor force, total`,
    net_migration           = `Net migration`,
    oop_exp_pc              = `Out-of-pocket expenditure per capita (current US$)`,
    access_water_basic      = `People using at least basic drinking water services (% of population)`,
    access_sanitation       = `People using at least basic sanitation services (% of population)`,
    pop_growth              = `Population growth (annual %)`,
    population_total        = `Population, total`,
    overweight_rate         = `Prevalence of overweight (% of adults)`,
    hypertension_rate       = `Prevalence of hypertension (% of adults ages 30-79)`,
    rural_pop               = `Rural population`,
    sex_ratio_birth         = `Sex ratio at birth (male births per female births)`,
    alcohol_pc              = `Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)`,
    tb_death_rate           = `Tuberculosis death rate (per 100,000 people)`,
    tb_treatment_success    = `Tuberculosis treatment success rate (% of new cases)`,
    unemployment_rate       = `Unemployment, total (% of total labor force)`,
    urban_pop_growth        = `Urban population growth (annual %)`,
    urban_pop_total         = `Urban population`,
    id_column               = ...1,
    life_exp                = `Life expectancy at birth, total (years)`) %>%
  dplyr::select(-Country, -id_column)
colSums(is.na(data_full))

# Correlation
cor_matrix <- cor(data_full %>% dplyr::select(where(is.numeric)), use = "pairwise.complete.obs")
strong_vars <- unique(rownames(cor_matrix)[apply(abs(cor_matrix) > 0.7 & abs(cor_matrix) < 1, 1, any)])
cor_strong <- cor_matrix[strong_vars, strong_vars]
png("correlation_plot_strong.png", width = 2000, height = 1600, res = 300)
corrplot::corrplot(cor_strong, method = "color", type = "upper",
                   tl.col = "black", tl.cex = 0.9, number.cex = 0.7,
                   col = colorRampPalette(c("red", "white", "blue"))(200),
                   mar = c(0, 0, 2, 0), title = "Variables with |Correlation| > 0.7")
dev.off()
# Check the distibution
p_hist <- data_full %>% pivot_longer(-life_exp, names_to = "Variable", values_to = "Value") %>% ggplot(aes(x = Value)) + 
  geom_histogram(bins = 25, fill = "skyblue", color = "white") + 
  facet_wrap(~Variable, scales = "free") + 
  theme_minimal()
ggsave("variable_distributions.png", plot = p_hist, width = 12, height = 10, dpi = 300)
# Check outliers
p_box <- data_full %>%
  pivot_longer(-life_exp, names_to = "Variable", values_to = "Value") %>%
  ggplot(aes(x = "", y = Value)) +
  geom_boxplot(fill = "lightblue", outlier.color = "red") +
  facet_wrap(~Variable, scales = "free") +
  theme_bw() +
  labs(title = "Boxplots for Outlier Detection")
ggsave("outlier_boxplots.png", plot = p_box,
       width = 12, height = 10, dpi = 300)
# ---------------- Helper Function: Preprocessing & Train/Test Split ----------------
to_log <- c("gni_pc", "gov_health_exp_pc", "health_exp_pc", "priv_health_exp_pc", "oop_exp_pc", "population_total", "rural_pop", "urban_pop_total", "labor_force_total", "net_migration", "tb_incidence", "tb_death_rate", "access_sanitation", "access_water_basic", "diabetes_rate", "fertility_rate")
set.seed(123)
idx_global <- createDataPartition(data_full$life_exp, p = 0.8, list = FALSE)
train_global <- data_full[idx_global, ]
test_global  <- data_full[-idx_global, ]

preprocess_data <- function(train_raw, test_raw, to_log, target = "life_exp") {
  # Impute medians on training only
  train_medians <- train_raw %>% summarise(across(where(is.numeric), ~ median(.x, na.rm = TRUE)))
  impute_fun <- function(df) mutate(df, across(where(is.numeric), ~ ifelse(is.na(.x), train_medians[[cur_column()]], .x)))
  train <- impute_fun(train_raw)
  test  <- impute_fun(test_raw)
  
  # Log transform (training reference)
  log_fun <- function(df, ref) mutate(
    df,
    across(all_of(to_log),
           ~ log1p(.x - min(ref[[cur_column()]], na.rm = TRUE) + 1))
  )
  train <- log_fun(train, train)
  test  <- log_fun(test, train)
  # Scale predictors only
  pred_cols <- setdiff(names(train), target)
  scale_fun <- function(df, ref) mutate(
    df,
    across(all_of(pred_cols),
           ~ (.x - mean(ref[[cur_column()]], na.rm = TRUE)) /
             sd(ref[[cur_column()]], na.rm = TRUE))
  )
  train_scaled <- scale_fun(train, train)
  test_scaled  <- scale_fun(test, train)
  # Reattach target (unscaled)
  train_scaled[[target]] <- train[[target]]
  test_scaled[[target]]  <- test[[target]]
  list(train = train_scaled, test = test_scaled)}
splits_all <- preprocess_data(train_global, test_global, to_log)
train_all <- splits_all$train
test_all  <- splits_all$test
# ------------------- BASELINE: OLS  -------------------
ols_full <- lm(life_exp ~ ., data = train_all)
ols_step <- stepAIC(ols_full, direction = "both", trace = FALSE)
summary(ols_step)
vif(ols_step)
# Evaluate predictive performance
step_pred  <- predict(ols_step, newdata = test_all)
step_resid <- test_all$life_exp - step_pred
# --- OLS Diagnostics ---
png("ols_residual_diagnostics.png", width = 2000, height = 1000, res = 300)
par(mfrow = c(1,2))
qqnorm(step_resid); qqline(step_resid, col="red")
plot(step_pred, step_resid, main="Residuals vs Fitted", xlab="Fitted", ylab="Residuals")
abline(h=0, col="red")
par(mfrow = c(1,1))
dev.off()
step_metrics <- list(
  rmse = sqrt(mean(step_resid^2)),
  mae  = mean(abs(step_resid)),
  r2   = 1 - sum(step_resid^2) /
    sum((test_all$life_exp - mean(test_all$life_exp))^2))
print(step_metrics)
set.seed(123)
cv_ols_step <- train(life_exp ~ ., data = train_all[, c(all.vars(formula(ols_step)))],
                     method = "lm", trControl = trainControl(method = "cv", number = 10))

# ------------------- PCA & PRINCIPAL COMPONENT REGRESSION ------------------
X_train <- train_all %>% dplyr::select(-life_exp)
y_train <- train_all$life_exp
X_test  <- test_all %>% dplyr::select(-life_exp)
y_test  <- test_all$life_exp

X_train <- X_train[, sapply(X_train, function(x) sd(x, na.rm = TRUE) > 0)]
X_test  <- X_test[, names(X_train)]
X_train[is.na(X_train)] <- 0
X_test[is.na(X_test)] <- 0

res_pca <- princomp(X_train, cor = TRUE, scores = TRUE)

p_scree <- fviz_eig(res_pca, addlabels = TRUE, main = "Scree Plot - PCA (Training Data)")
ggsave("pca_scree_plot_corrected.png", plot = p_scree, width = 8, height = 6, dpi = 300)

# --- 1. Permutation Test (compare observed vs random eigenvalues) ---
permtestPCA <- function(data, nperm = 1000) {
  set.seed(123)
  eigen_real <- eigen(cor(data, use = "pairwise.complete.obs"))$values
  eigen_perm <- replicate(nperm, {
    perm_data <- apply(data, 2, sample)
    eigen(cor(perm_data, use = "pairwise.complete.obs"))$values
  })
  ci_lower <- apply(eigen_perm, 1, quantile, 0.025)
  ci_upper <- apply(eigen_perm, 1, quantile, 0.975)
  
  df_perm <- data.frame(
    Component = 1:length(eigen_real),
    Observed = eigen_real,
    Lower = ci_lower,
    Upper = ci_upper)
  
  ggplot(df_perm, aes(x = Component)) +
    geom_line(aes(y = Observed, color = "Observed"), linewidth = 1) +
    geom_point(aes(y = Observed, color = "Observed"), size = 2) +
    geom_line(aes(y = Lower, color = "2.5% CI"), linetype = "dashed") +
    geom_line(aes(y = Upper, color = "97.5% CI"), linetype = "dashed") +
    scale_color_manual(values = c("Observed" = "red", "2.5% CI" = "blue", "97.5% CI" = "blue")) +
    labs(title = "Permutation Test PCA", x = "Component", y = "Eigenvalue", color = "") +
    theme_minimal(base_size = 13)}
set.seed(123)
p_perm <- permtestPCA(X_train)
ggsave("pca_permutation_test.png", plot = p_perm, width = 8, height = 6, dpi = 300)

# --- 2. Bootstrap Test for Eigenvalues (stability + Kaiser rule) ---
my_boot_pca <- function(x, ind) {
  res <- princomp(x[ind, ], cor = TRUE)
  res$sdev^2
}
set.seed(123)
fit.boot <- boot(data = X_train, statistic = my_boot_pca, R = 1000)
eigs.boot <- fit.boot$t
obs_eigs <- res_pca$sdev^2
# Plot bootstrap distributions
png("pca_bootstrap_boxplot.png", width = 2000, height = 1600, res = 300)
boxplot(eigs.boot, col = "beige", las = 1,
        main = "Bootstrap Distribution of Eigenvalues",
        xlab = "Principal Component", ylab = "Eigenvalue")
points(colMeans(eigs.boot), pch = 19, col = "darkblue")
abline(h = 1, col = "red", lty = 2)
dev.off()
# --- Histogram of Bootstrap Distribution for the First Eigenvalue ---
png("pca_bootstrap_histogram_pc1.png", width = 2000, height = 1600, res = 300)
hist(eigs.boot[, 1],
     xlab = "Eigenvalue 1",
     main = "Bootstrap Distribution of Eigenvalue 1",
     col = "lightblue", border = "white",
     las = 1, breaks = 25)
perc.alpha <- quantile(eigs.boot[, 1], c(0.025, 0.975))
abline(v = perc.alpha, col = "green", lwd = 2)
abline(v = obs_eigs[1], col = "red", lwd = 2)
legend("topright", legend = c("Observed", "95% CI limits"),
       col = c("red", "green"), lty = 1, lwd = 2, bty = "n")
dev.off()

# Compute 95% CI for each eigenvalue
ci_eigs <- apply(eigs.boot, 2, quantile, c(0.025, 0.975))
ci_table <- data.frame(
  Component = 1:length(obs_eigs),
  Observed = round(obs_eigs, 3),
  CI_Lower = round(ci_eigs[1, ], 3),
  CI_Upper = round(ci_eigs[2, ], 3))
print(ci_table)

# --- 3. Bootstrap Test for total variance ≥ 70% (VAF test) ---
boot_vaf <- apply(eigs.boot, 1, function(x) sum(x[1:5]) / sum(x))
ci_vaf <- quantile(boot_vaf, c(0.025, 0.975))
cat("95% Bootstrap CI for cumulative variance (5 PCs):", round(ci_vaf, 3), "\n")

# --- 4. Decide number of components to retain ---
n_components <- sum(ci_eigs[1, ] > 1)   
n_components <- 5
# Project data onto the first 5 PCs
pc_train <- predict(res_pca, newdata = X_train)
pc_test  <- predict(res_pca, newdata = X_test)
train_scores <- as.data.frame(pc_train[, 1:n_components, drop = FALSE])
test_scores  <- as.data.frame(pc_test[, 1:n_components, drop = FALSE])
train_scores$life_exp <- y_train
test_scores$life_exp  <- y_test
# Fit regression on principal components
pcr_model <- lm(life_exp ~ ., data = train_scores)
pcr_pred <- predict(pcr_model, newdata = test_scores)
pcr_resid <- test_scores$life_exp - pcr_pred
# Performance metrics
pcr_rmse <- sqrt(mean(pcr_resid^2))
pcr_mae  <- mean(abs(pcr_resid))
pcr_r2   <- 1 - sum(pcr_resid^2) / sum((y_test - mean(y_test))^2)
cat("PCR (5 PCs): RMSE =", round(pcr_rmse, 3), 
    "MAE =", round(pcr_mae, 3), 
    "R² =", round(pcr_r2, 3), "\n")
# Variable contribution visualization
p_pca_var <- fviz_pca_var(
  res_pca,
  col.var = "contrib",
  gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"),
  repel = TRUE,
  title = "PCA - Variable Contributions")
ggsave("pca_variable_contributions.png", plot = p_pca_var, width = 8, height = 6, dpi = 300)
# Top contributing variables per component (PC1–PC5)
loadings_df <- as.data.frame(res_pca$loadings[, 1:n_components]) %>%
  rownames_to_column("Variable") %>%
  pivot_longer(-Variable, names_to = "Component", values_to = "Loading") %>%
  mutate(AbsLoading = abs(Loading)) %>%
  group_by(Component) %>%
  slice_max(order_by = AbsLoading, n = 10) %>%
  ungroup()

p_top_vars <- ggplot(loadings_df, aes(x = reorder(Variable, AbsLoading), 
                                      y = AbsLoading, fill = Component)) +
  geom_col(show.legend = TRUE) +
  coord_flip() +
  facet_wrap(~Component, scales = "free_y") +
  labs(title = "Top 10 Most Influential Variables per Principal Component",
       x = "Variable", y = "Absolute Loading") +
  theme_minimal(base_size = 12)
ggsave("pca_top_variable_contributions.png", plot = p_top_vars, width = 10, height = 8, dpi = 300)

# ---------------- Elastic Net ----------------------
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
alpha_grid <- seq(0, 1, length = 10)
lambda_seq <- 10^seq(2, -3, length = 100)
set.seed(123)
elasticNet <- train(
  life_exp ~ ., data = train_all,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = alpha_grid, lambda = lambda_seq),
  trControl = ctrl, standardize = TRUE)

best <- elasticNet$bestTune
final_en <- glmnet(
  x = as.matrix(dplyr::select(train_all, -life_exp)),
  y = train_all$life_exp,
  alpha = best$alpha, lambda = best$lambda, standardize = TRUE)
pred_en <- as.numeric(predict(final_en, newx = as.matrix(dplyr::select(test_all, -life_exp))))
enet_resid <- test_all$life_exp - pred_en  
print(best$alpha)
print(best$lambda)

en_metrics <- list(
  rmse = sqrt(mean(enet_resid^2)),
  mae  = mean(abs(enet_resid)),
  r2   = 1 - sum(enet_resid^2) / sum((test_all$life_exp - mean(test_all$life_exp))^2)
)
print(en_metrics)
# View important variables
enet_coef <- coef(final_en)
coef_df <- data.frame(
  Variable = rownames(enet_coef),
  Coefficient = as.numeric(enet_coef)
) %>%
  filter(Coefficient != 0 & Variable != "(Intercept)") %>%
  arrange(desc(abs(Coefficient)))
# Plot coefficients
p_coef <- ggplot(coef_df, aes(x = reorder(Variable, Coefficient), y = Coefficient)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(title = "Elastic Net Non-Zero Coefficients",
       y = "Coefficient", x = "")
ggsave("elasticnet_coefficients.png", plot = p_coef, width = 10, height = 8, dpi = 300)

# ------------------- MODEL PERFORMANCE COMPARISON -------------------
comparison <- tibble(
  Model = c("OLS (Interaction)", "PCR (5 PCs)", "Elastic Net"),
  RMSE  = c(step_metrics$rmse, pcr_rmse, en_metrics$rmse),
  MAE   = c(step_metrics$mae,  pcr_mae,  en_metrics$mae),
  R2    = c(step_metrics$r2,   pcr_r2,   en_metrics$r2)
) %>%
  mutate(across(where(is.numeric), ~ round(.x, 4)))
# Print comparison table
print(comparison)
# ------------------- RESIDUAL DIAGNOSTICS -------------------
plot_residuals <- function(fitted, resid, title, color) {
  df <- data.frame(Fitted = fitted, Residuals = resid)
  p1 <- ggplot(df, aes(Fitted, Residuals)) +
    geom_point(alpha = 0.6, color = color) +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = paste(title, "Residuals vs Fitted")) +
    theme_minimal()
  p2 <- ggplot(df, aes(Residuals)) +
    geom_histogram(bins = 25, fill = color, color = "white") +
    labs(title = paste(title, "Residual Distribution")) +
    theme_minimal() 
  p1 / p2}
# Generate plots for all models in one go
resid_plots <- list(
  OLS = plot_residuals(step_pred, test_all$life_exp - step_pred, "OLS", "steelblue"),
  PCR = plot_residuals(fitted(pcr_model), residuals(pcr_model), "PCR", "orange"),
  EN  = plot_residuals(as.numeric(pred_en), 
                       as.numeric(test_all$life_exp - pred_en),
                       "Elastic Net", "firebrick"))
# Combine all models side by side
final_resid_plot <- resid_plots$OLS | resid_plots$PCR | resid_plots$EN +
  plot_annotation(title = "Residual Diagnostics: OLS vs PCR vs Elastic Net",
                  theme = theme(plot.title = element_text(size = 16, face = "bold")))
ggsave("residual_diagnostics.png", plot = final_resid_plot,
       width = 14, height = 6, dpi = 300)
# --------- Predicting on unseen data -------
predictions <- predictions %>%
  dplyr::rename(
    fertility_rate          = `Fertility rate, total (births per woman)`,
    health_exp_pc           = `Current health expenditure per capita (current US$)`,
    diabetes_rate           = `Diabetes prevalence (% of population ages 20 to 79)`,
    gov_health_exp_pc       = `Domestic general government health expenditure per capita (current US$)`,
    priv_health_exp_pc      = `Domestic private health expenditure per capita (current US$)`,
    gni_pc                  = `GNI per capita, Atlas method (current US$)`,
    immun_dpt               = `Immunization, DPT (% of children ages 12-23 months)`,
    immun_hepb3             = `Immunization, HepB3 (% of one-year-old children)`,
    immun_hib3              = `Immunization, Hib3 (% of children ages 12-23 months)`,
    immun_measles           = `Immunization, measles (% of children ages 12-23 months)`,
    immun_pol3              = `Immunization, Pol3 (% of one-year-old children)`,
    tb_incidence            = `Incidence of tuberculosis (per 100,000 people)`,
    labor_force_total       = `Labor force, total`,
    net_migration           = `Net migration`,
    oop_exp_pc              = `Out-of-pocket expenditure per capita (current US$)`,
    access_water_basic      = `People using at least basic drinking water services (% of population)`,
    access_sanitation       = `People using at least basic sanitation services (% of population)`,
    pop_growth              = `Population growth (annual %)`,
    population_total        = `Population, total`,
    overweight_rate         = `Prevalence of overweight (% of adults)`,
    hypertension_rate       = `Prevalence of hypertension (% of adults ages 30-79)`,
    rural_pop               = `Rural population`,
    sex_ratio_birth         = `Sex ratio at birth (male births per female births)`,
    alcohol_pc              = `Total alcohol consumption per capita (liters of pure alcohol, projected estimates, 15+ years of age)`,
    tb_death_rate           = `Tuberculosis death rate (per 100,000 people)`,
    tb_treatment_success    = `Tuberculosis treatment success rate (% of new cases)`,
    unemployment_rate       = `Unemployment, total (% of total labor force)`,
    urban_pop_growth        = `Urban population growth (annual %)`,
    urban_pop_total         = `Urban population`) %>%
  dplyr::select(-...1)


new_data_split <- preprocess_data(train_global, predictions, to_log)
pred_final <- new_data_split$test
pred_final_aligned <- pred_final[, names(dplyr::select(train_all, -life_exp))]
final_predictions <- predict(
  final_en,
  newx = as.matrix(pred_final_aligned),
  s = best$lambda)
predicted_df <- data.frame(
  Country = predictions$Country,
  Predicted_Life_Expectancy = round(as.numeric(final_predictions), 1))
print(predicted_df)
```