## Loading packages and data
pacman::p_load(
  tidyverse,
  ggplot2,
  dplyr,
  car,
  caret,
  caretEnsemble,
  elasticnet,
  glmnet,
  broom,
  psych,
  corrplot
)

# Load the Dataset
df <- read_csv("Documents/DSMA Masters/BLOK1/Intro to Data Science/Group Project Intro to Data Science/a1_data_group_2.csv")
head(df, 10)

df <- df %>%
  dplyr::rename(
    country                = `Country`,
    electricity_access     = `Access to electricity (% of population)`,
    adolescent_fertility   = `Adolescent fertility rate (births per 1,000 women ages 15-19)`,
    age_dependency   = `Age dependency ratio (% of working-age population)`,
    contrib_family_fem     = `Contributing family workers, female (% of female employment) (modeled ILO estimate)`,
    contrib_family_male    = `Contributing family workers, male (% of male employment) (modeled ILO estimate)`,
    contrib_family_total   = `Contributing family workers, total (% of total employment) (modeled ILO estimate)`,
    credit_info_index      = `Depth of credit information index (0=low to 8=high)`,
    employers_fem          = `Employers, female (% of female employment) (modeled ILO estimate)`,
    employers_male         = `Employers, male (% of male employment) (modeled ILO estimate)`,
    employers_total        = `Employers, total (% of total employment) (modeled ILO estimate)`,
    emp_agriculture_total  = `Employment in agriculture (% of total employment) (modeled ILO estimate)`,
    emp_agriculture_fem    = `Employment in agriculture, female (% of female employment) (modeled ILO estimate)`,
    emp_agriculture_male   = `Employment in agriculture, male (% of male employment) (modeled ILO estimate)`,
    emp_industry_total     = `Employment in industry (% of total employment) (modeled ILO estimate)`,
    emp_industry_fem       = `Employment in industry, female (% of female employment) (modeled ILO estimate)`,
    emp_industry_male      = `Employment in industry, male (% of male employment) (modeled ILO estimate)`,
    emp_services_total     = `Employment in services (% of total employment) (modeled ILO estimate)`,
    emp_services_fem       = `Employment in services, female (% of female employment) (modeled ILO estimate)`,
    emp_services_male      = `Employment in services, male (% of male employment) (modeled ILO estimate)`,
    export_value_index     = `Export value index (2000 = 100)`,
    export_volume_index    = `Export volume index (2000 = 100)`,
    fertility_rate_total   = `Fertility rate, total (births per woman)`,
    broadband_subs         = `Fixed broadband Internet subscribers (per 100 people)`,
    gdp_growth             = `GDP growth (annual %)`,
    gdp_pc_const2005       = `GDP per capita (constant 2005 US$)`,
    gdp_pc_ppp             = `GDP per capita, PPP (constant 2011 international $)`,
    internet_users         = `Individuals using the Internet (% of population)`,
    lfpr_fem               = `Labor force participation rate, female (% of female population ages 15+) (modeled ILO estimate)`,
    lfpr_male              = `Labor force participation rate, male (% of male population ages 15+) (modeled ILO estimate)`,
    lfpr_total             = `Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)`,
    labor_force_total      = `Labor force, total`,
    life_expectancy_female           = `Life expectancy at birth, female (years)`,
    life_expectancy_male          = `Life expectancy at birth, male (years)`,
    mobile_subs            = `Mobile cellular subscriptions (per 100 people)`,
    own_account_fem        = `Own-account workers, female (% of female employment) (modeled ILO estimate)`,
    own_account_male       = `Own-account workers, male (% of male employment) (modeled ILO estimate)`,
    own_account_total      = `Own-account workers, total (% of male employment) (modeled ILO estimate)`,
    pop_0_14_pct           = `Population ages 0-14 (% of total)`,
    pop_0_14_total         = `Population ages 0-14, total`,
    pop_15_64_pct          = `Population ages 15-64 (% of total)`,
    pop_15_64_total        = `Population ages 15-64, total`,
    pop_65plus_pct         = `Population ages 65 and above (% of total)`,
    pop_65plus_total       = `Population ages 65 and above, total`,
    pop_density            = `Population density (people per sq. km of land area)`,
    population_growth             = `Population growth (annual %)`,
    pop_total              = `Population, total`,
    credit_coverage_priv   = `Private credit bureau coverage (% of adults)`,
    credit_coverage_pub    = `Public credit registry coverage (% of adults)`,
    rural_pop_total        = `Rural population`,
    rural_pop_pct          = `Rural population (% of total population)`,
    self_emp_fem           = `Self-employed, female (% of female employment) (modeled ILO estimate)`,
    self_emp_male          = `Self-employed, male (% of male employment) (modeled ILO estimate)`,
    self_emp_total         = `Self-employed, total (% of total employment) (modeled ILO estimate)`,
    tax_payments       = `Tax payments (number)`,
    telephone_lines        = `Telephone lines (per 100 people)`,
    contract_days          = `Time required to enforce a contract (days)`,
    start_business_days    = `Time required to start a business (days)`,
    tax_prep_hours         = `Time to prepare and pay taxes (hours)`,
    unemployment_fem              = `Unemployment, female (% of female labor force) (modeled ILO estimate)`,
    unemployment_male             = `Unemployment, male (% of male labor force) (modeled ILO estimate)`,
    unemployment_total            = `Unemployment, total (% of total labor force) (modeled ILO estimate)`,
    unemp_youth_fem        = `Unemployment, youth female (% of female labor force ages 15-24) (modeled ILO estimate)`,
    unemp_youth_male       = `Unemployment, youth male (% of male labor force ages 15-24) (modeled ILO estimate)`,
    unemp_youth_total      = `Unemployment, youth total (% of total labor force ages 15-24) (modeled ILO estimate)`,
    urban_pop_total        = `Urban population`,
    urban_pct          = `Urban population (% of total)`,
    vulner_emp_fem         = `Vulnerable employment, female (% of female employment) (modeled ILO estimate)`,
    vulner_emp_male        = `Vulnerable employment, male (% of male employment) (modeled ILO estimate)`,
    vulner_emp_total       = `Vulnerable employment, total (% of total employment) (modeled ILO estimate)`,
    waged_emp_fem          = `Wage and salaried workers, female (% of female employment) (modeled ILO estimate)`,
    waged_emp_male         = `Wage and salaried workers, male (% of male employment) (modeled ILO estimate)`,
    waged_emp_total        = `Wage and salaried workers, total (% of total employment) (modeled ILO estimate)`,
    net_trade              = `Net trade in goods and services (BoP, current US$)`
  ) 


# 1.1 Select & rename variables used in models -> to the report add a sentance about selected variables logic
model_df_linear <- df %>%
  dplyr::select(
    country,           
    gdp_growth,         
    gdp_pc_ppp,        
    net_trade,         
    unemployment_total, 
    age_dependency,     
    urban_pct,   
    life_expectancy_female, 
    tax_payments, 
    population_growth 
  )

# 1.2 Basic checks
colSums(is.na(df)) %>% sort(decreasing = TRUE) # no missing values
sum(duplicated(model_df_linear$country))
str(model_df_linear)

# 1.3 Check for outlieres
summary(model_df_linear)
boxplot(model_df_linear$gdp_pc_ppp, main="GDP per capita PPP")
boxplot(model_df_linear$net_trade, main="Net trade")

###There are two visible outliers (eg. extreme oil exporters or countries Luxemburg with extremely high GDP per capita )
### Keep in mind for OLS and regression interpretation


# 1.5 Check distribution of outcome
hist(model_df_linear$gdp_growth, 
     main="GDP Growth Distribution", 
     xlab="Growth (%)", 
     ylab = "Frequency",
     col = "lightgray", border = "black")
qqnorm(model_df_linear$gdp_growth); qqline(model_df_linear$gdp_growth, col="red")

----------------------------------------------
  
  # 2. Descriptive Statistics
  
  # Summary stats for linear model dataset
  summary_stats <- psych::describe(model_df_linear %>% 
                                     dplyr::select(-country))
summary_stats %>% 
  dplyr::select(mean, sd, min, max, skew, kurtosis)


# Correlation Heatmap
cor_matrix <- cor(model_df_linear %>% dplyr::select(-country), use = "pairwise.complete.obs")

corrplot(cor_matrix, method = "color", type = "upper",
         tl.col = "black", tl.srt = 45, 
         addCoef.col = "black", number.cex = 0.6,
         title = "Correlation Heatmap of Predictors & GDP Growth",
         mar=c(0,0,1,0))

# Scatterplots: GDP growth vs each predictor

long_df <- model_df_linear %>%
  pivot_longer(-c(country, gdp_growth), names_to = "variable", values_to = "value")

ggplot(long_df, aes(x = value, y = gdp_growth)) +
  geom_point(color = "steelblue", alpha = 0.7) +
  geom_smooth(method = "lm", se = FALSE, color = "red", linewidth = 0.7) +
  facet_wrap(~variable, scales = "free_x") +
  theme_minimal() +
  labs(title = "GDP Growth vs Predictors",
       x = "Predictor Value", y = "GDP Growth (%)")

# - For most predictors, GDP growth appears evenly distributed across the variable range, 
# with no strong linear trend. This suggests that growth is not explained by simple bivariate
# linear relationships for these predictors.
# 
# - Net trade stands out: while most countries trade close to balance, a handful 
# have very large surpluses or deficits, creating extreme outliers. This skews the 
# distribution and explains why net trade does not show a strong linear relationship with GDP growth.”
# 
# - These patterns highlight the importance of multivariate modeling: even if predictors 
# individually show weak associations, they may collectively explain variance in GDP growth 
# once considered together.

# Q1: Linear Regression Prediction

# Model A: baseline regression without Net trade

formula_A <- gdp_growth ~ gdp_pc_ppp + unemployment_total +
  age_dependency + urban_pct +
  life_expectancy_female + tax_payments + population_growth

model_A <- lm(formula_A, data = model_df_linear, singular.ok = FALSE)


# Model B: regression with Net Trade 
formula_B <- gdp_growth ~ gdp_pc_ppp + unemployment_total +
  age_dependency + urban_pct +
  life_expectancy_female + tax_payments +
  net_trade + population_growth

model_B <- lm(formula_B, data = model_df_linear, singular.ok = FALSE)

summary(model_A)
summary(model_B)

# summary(model_A)$adj.r.squared
# [1] 0.01383 
# summary(model_B)$adj.r.squared
# [1] 0.00724 

## We ran a linear regression to see if there is significant correlation, and possibly
## causation if assumptions hold, between different aspects such as Net Trade in goods and services, 
## and a load of other variables plausibly linked with the health of an economy, such as 
## gdp per capita in PPP terms, measured in constant 2011 US dollar equivalents. unemployment,
## life expectancy, urban population rate, population growth, age dependency ratio and the 
## amount of tax payments. We included Net Trade, as Net Trade is one of the sources of GDP, so 
## higher Net Trade would be expected to have a positive influence on GDP growth. When 
## looking at the output of the regression, we see that none of the factors has a 
## significant correlation with GDP growth. This is also the case for Net Trade, so including
## Net Trade has not been helpful in predicting GDP growth. Also, the adjusted R-squared  
## decreased, so the model decreased in explanative capability. This is because the variation in
## Net Trade is too big for the estimated positive effect to be surely caused by an actual 
## causal influence, instead being plausibly caused just by chance. Therefore, from
## now on we drop Net trade as a factor in our analysis. 

## Task 2:

formula_C <- gdp_growth ~ gdp_pc_ppp + unemployment_total +
  age_dependency + I(unemployment_total^2) + urban_pct +
  life_expectancy_female + population_growth+ tax_payments

model_C <- lm(formula_C, data = model_df_linear, singular.ok = FALSE)
summary(model_C)

### Adj. R sqr = 0.03226 
## Higher R² when adding a squared factor for unemployment, and the unemployment variable
## is significant in both the unemployment and unemployment². Therefore, including
## the polynomial results in better fit and should be included. However, before interpreting
## the coefficients, we run OLS diagnostics to compare both models in this regard.

## model diagnostics model A (without polynomial term and trade)
plot(model_A)
## does not seem to be non-linearity and heteroskedasticity in residual plot, but there 
## is an outlier in fitted values and qqplot graphs. Also, qq-plot slight deviations 
## around the tails but for the rest fitted quite nicely on teh line. 
## no high residual in high-leverage point --> flat line in residuals vs leverage
vif(model_A)
## no multicollinearity

## model diagnostics model 2
plot(model_C)
## plots look the same for the model in question 2 than in question 3 --> do not seem to have
## a problem with non-linearity and heteroskedasticity, there is an outlier. qq-plot 
## slight deviations around the tails but for the rest fitted quite nicely on teh line. 
## no high residual in high-leverage point --> flat line in residuals vs leverage
vif(model_C)
## multicollinearity between unemployment and unemployment squared, as expected. 
## This is not necessarily an issue, since they are both of the same variable, and
## can be used for model prediction. However, independent interpretation of both 
## seperate from one another is not ideal.

## the model diagnostics do not significantly change when including the polynomial,
## and look decent either way. 

## Task 3: Now run a penalized regression using LASSO to check if the model works. Do not 
## mind adjustments at this stage, just run the model with the same variables as the 
## regression you ran in the question before. Does the model work? Explain what is the 
## difference in comparison to 2.

X <- model.matrix(formula_C, data = model_df_linear)[, -1]
y <- model_df_linear$gdp_growth


set.seed(555)

cvfit <- cv.glmnet(x=X, y=y, alpha=1, type.measure = "mse", nfolds = 10)
print(cvfit)

plot(cvfit)

coef(cvfit, s = "lambda.min")
coef(cvfit, s = "lambda.1se")

## the LASSO deleted most variables in the minimal lambda (lambda=0.2740), meaning that 
## the LASSO worked. With the lambda using 1 standard deviation away, all coefficients 
## are even deleted. 
## The difference between the LASSO and the model in question 2 is that
## LASSO penalizes the use of variables. Thereby, LASSO shrinks the coefficients with
## a penalization. Subsequently, it minimizes the sum of the residual sum of squares
## and the penalty, which means that the variables that decrease are the ones that 
## are actually important in predicting GDP growth. LASSO subsequently takes out
## variables that have a coefficient of 0. In our case, this happened with most variables,
## indicating that the variables do not perform well in predicting GDP growth.
## Now, the only variables that remained in the model were those of GDP per capita
## and Unemployment for the minimal lambda. The coefficients in the LASSO regression 
## were smaller than those for the linear regression, which makes sense, as LASSO 
## decreases coefficient size.

## Task 4: Why is it important to standardize predictors before applying penalized 
## regression? Demonstrate with one example from your dataset.

## It is important to standardize predictors before applying penalized regression, 
## because LASSO penalizes variables on their coefficient size. When predictors are
## not standardized beforehand, this means that the different variables get "judged"
## differently based on their scale, with a variable measured in cents being judged
## 100 times as much as a variable measured in dollars, or a variable in year-on-year
## change different than one showing total rates.

set.seed(555)

cvfit1 <- cv.glmnet(x=X, y=y, alpha=1, type.measure = "mse", nfolds = 10, standardize=TRUE)
plot(cvfit1$glmnet.fit, xvar = "lambda", label = TRUE)


cvfit2 <- cv.glmnet(x=X, y=y, alpha=1, type.measure = "mse", nfolds = 10, standardize=FALSE)
plot(cvfit2$glmnet.fit, xvar = "lambda", label = TRUE)

# Coefficeints of both models
coef(cvfit2, s = "lambda.min")
coef(cvfit1, s = "lambda.min")


## Rationale: Without standardization, variables measured in large numbers (like dollars) get punished more 
## heavily than those in percentages, not because they’re less important, but just because of their units. 
## Standardizing puts everything on the same scale so the algorithm can judge importance fairly.

## Task 5: After you show the results with LASSO, your manager is convinced that 
## penalized regression is the way to go. Hence, she asks you to construct two 
## additional penalized regression models to predict GDP growth (annual %).
## even teams work with ridge regression and elastic net --> next couple of questions

## Task 6: Randomly divide your dataset into training samples and test samples of 
## 110 and 40 observations, respectively
# Train/Test split (110/40)
set.seed(555)

n <- nrow(X)
idx <- sample(seq_len(n), size = 110)  

trainData <- df[idx,]
y2<-df$gdp_growth
X2 <- model.matrix(gdp_growth ~ . -country, data=df)[,-1]

X_train <- X2[idx, ]; y_train <- y2[idx]
X_test <- X2[-idx,]; y_test <- y2[-idx]


## Task 7: For the training sample, run both methods, while choosing the optimal penalty 
## parameter through crossvalidation (for elastic net, you should optimize both α and λ). 
## Report the final model for both methods (you can put the table in the appendix of the
## report). Give an interpretation, including which variables are more relevant for predicting 
## GPD growth, and point out similarities and differences between the two models that 
## you consider relevant.

## a) Ridge CV on training set
set.seed(555)
cv_ridge <- cv.glmnet(X_train, y_train, alpha = 0, nfolds = 10)

lambda_ridge_min <- cv_ridge$lambda.min
lambda_ridge_1se <- cv_ridge$lambda.1se

### Final ridge models at the two lambdas
ridge_min <- glmnet(X_train, y_train, alpha = 0, lambda = lambda_ridge_min)
ridge_1se <- glmnet(X_train, y_train, alpha = 0, lambda = lambda_ridge_1se)


## Elastic Net
set.seed(555)

# Cross-validation setup
fitControl <- trainControl(
  method = "repeatedcv",
  number = 10,
  repeats = 5,
  verboseIter = TRUE
)

# Candidate alpha values
alpha_grid <- seq(0.0001, 1, length = 10)

# Use lambda sequence from ridge CV fit earlier
lambda_seq <- cv_ridge$glmnet.fit$lambda  

# Run Elastic Net CV with caret
elasticNet <- train(
  gdp_growth ~ . - country,
  data = trainData,
  method = "glmnet",
  tuneGrid = expand.grid(alpha = alpha_grid,
                         lambda = lambda_seq),
  trControl = fitControl
)

# Show best tuning parameters
print(elasticNet$bestTune)

# Extract best alpha and lambda
best_alpha  <- elasticNet$bestTune$alpha
best_lambda <- elasticNet$bestTune$lambda

# Fit final Elastic Net model
en_final <- glmnet(X_train, y_train,
                   alpha = best_alpha,
                   lambda = best_lambda,
                   standardize = TRUE)

# Coefficients of the final model
coef(en_final)

## Interpretation: Francesco and Michael
## variable importance
plot(varImp(elasticNet, scale = FALSE))

## Task 8: Compare the quality of the predictions from both models (using an 
## appropriate metric). You will compute predictions using lambda min and lambda 1se 
## that you found thanks to cross-validation, such that you have 4 models. Interpret 
## the results

## Task 8: Compare the quality of the predictions from both models (using an 
## appropriate metric). You will compute predictions using lambda min and lambda 1se 
## that you found thanks to cross-validation, such that you have 4 models.

# Predictions for Ridge models
pred_ridge_min  <- predict(ridge_min, newx = X_test)
pred_ridge_1se  <- predict(ridge_1se, newx = X_test)

# Elastic Net models: λ.min and λ.1se
en_min <- glmnet(X_train, y_train,
                 alpha = best_alpha,
                 lambda = elasticNet$bestTune$lambda, # λ.min from caret
                 standardize = TRUE)

# For λ.1se, pick the largest lambda within 1 SE of min error
results_all <- elasticNet$results
min_rmse <- min(results_all$RMSE)
rmse_1se <- min_rmse + results_all$RMSESD[which.min(results_all$RMSE)]
lambda_en_1se <- max(results_all$lambda[results_all$RMSE <= rmse_1se])

en_1se <- glmnet(X_train, y_train,
                 alpha = best_alpha,
                 lambda = lambda_en_1se,
                 standardize = TRUE)

# Predictions for Elastic Net models
pred_en_min  <- predict(en_min, newx = X_test)
pred_en_1se  <- predict(en_1se, newx = X_test)

# Performance metric functions
rmse <- function(y, yhat) sqrt(mean((y - as.numeric(yhat))^2))
mae  <- function(y, yhat) mean(abs(y - as.numeric(yhat)))

# Collect results in one table
results <- tibble(
  model = c(
    "Ridge (lambda.min)",
    "Ridge (lambda.1se)",
    paste0("Elastic Net (alpha=", round(best_alpha, 2), ", lambda.min)"),
    paste0("Elastic Net (alpha=", round(best_alpha, 2), ", lambda.1se)")
  ),
  RMSE = c(
    rmse(y_test, pred_ridge_min),
    rmse(y_test, pred_ridge_1se),
    rmse(y_test, pred_en_min),
    rmse(y_test, pred_en_1se)
  ),
  MAE = c(
    mae(y_test, pred_ridge_min),
    mae(y_test, pred_ridge_1se),
    mae(y_test, pred_en_min),
    mae(y_test, pred_en_1se)
  )
)

print(results)

## interpretation -> Francesco and Michael

## Task 9: Discuss why choosing λ purely based on minimizing cross-validation error 
## may sometimes lead to overfitting. What is the role of the “1-SE rule” in mitigating
## this risk?

## Choosing λ purely based on minimizing cross-validation error may lead to overfitting,
## because the Ridge/elastic net models estimate lambda using solely the training data. 
## therefore, it might bias the estimators to such an extent that it predicts the outcome
## in the training dataset with great accuracy, but performs less well in predicting
## "new" outcomes in the testing dataset through overfitting. Therefore, the 1-SE rule
## can mitigate this risk, as it penalizes the coefficients further, which mean further
## shrinking of the coefficients, which are therefore less biased in order to fit the 
## training data too perfectly, leading to less overfitting. Therefore, it is good 
## practice to test both the ideal lambda and the lambda using the "1-SE rule"

## Task 10: INTERPRETATION --> MICHAEL AND FRANCESCO

## Task 11: It might also be interesting to consider if a country grows more or less 
## than most other countries. Create a new variable that classifies a country as 
## Growing more in which you record values of GDP growth above 2.7% as 1 and below
## 2.7% as 0.
df <- df %>%
  mutate(Growing_more = ifelse(gdp_growth > 2.7, 1, 0))

# Check up frequency table
table(df$Growing_more)


# Binary dependent Variable witg logistic Ridge regression 
  
#Create new matrix with new dependent variable
X3 <- model.matrix(Growing_more ~ . - country, data=df)[,-1]
y3 <- df$Growing_more

set.seed(555)
n <- nrow(X3)

idx1 <- sample(seq_len(n), size = 110)  # training set
X_train1 <- X3[idx1, ]; y_train1 <- y3[idx1]
X_test1  <- X3[-idx1,]; y_test1  <- y3[-idx1]

set.seed(555)
cv_ridge_logit <- cv.glmnet(X_train1, y_train1, alpha = 0, family = "binomial", nfolds = 10)

lambda_min <- cv_ridge_logit$lambda.min
lambda_1se <- cv_ridge_logit$lambda.1se

ridge_logit_min <- glmnet(X_train1, y_train1, 
                          alpha = 0, 
                          lambda = lambda_min, 
                          family = "binomial", 
                          nfold = 10)

ridge_logit_1se <- glmnet(X_train1, y_train1, 
                          alpha = 0, 
                          lambda = lambda_1se, 
                          family = "binomial", 
                          nfold = 10)

# Probabilities
prob_min  <- predict(ridge_logit_min, newx = X_test1, type = "response")
prob_1se  <- predict(ridge_logit_1se, newx = X_test1, type = "response")

# Class predictions (threshold 0.5)
pred_min <- ifelse(prob_min > 0.5, 1, 0)
pred_1se <- ifelse(prob_1se > 0.5, 1, 0)

# Accuracy
acc_min  <- mean(pred_min == y_test1)
acc_1se  <- mean(pred_1se == y_test1)

# Confusion matrices
table(Predicted = pred_min, Actual = y_test1)
acc_min
table(Predicted = pred_1se, Actual = y_test1)
acc_1se

## Interpretation : 
# At λmin:
# True Negatives (TN) = 11 (predicted 0, actual 0)
# False Negatives (FN) = 3 (predicted 0, actual 1)
# False Positives (FP) = 6 (predicted 1, actual 0)
# True Positives (TP) = 20 (predicted 1, actual 1)
## overall accuracy: 77.5%

# At λ1se:
# TN = 11
# FN = 3
# FP = 8
# TP = 18
## overall accuracy: 72.5%


## Different Data Split
set.seed(555)
n <- nrow(X3)

idx2 <- sample(seq_len(n), size = 100)

X_train2 <- X3[idx2, ]; y_train2 <- y3[idx2]
X_test2  <- X3[-idx2,]; y_test2  <- y3[-idx2]

# Cross-validated logistic Ridge
cv_ridge_logit2 <- cv.glmnet(X_train2, y_train2,
                             alpha = 0,
                             family = "binomial",
                             nfolds = 10)

lambda_min2 <- cv_ridge_logit2$lambda.min
lambda_1se2 <- cv_ridge_logit2$lambda.1se

ridge_logit_min2 <- glmnet(X_train2, y_train2, alpha = 0, lambda = lambda_min2, family = "binomial")
ridge_logit_1se2 <- glmnet(X_train2, y_train2, alpha = 0, lambda = lambda_1se2, family = "binomial")

# Predictions on new test set
prob_min2 <- predict(ridge_logit_min2, newx = X_test2, type = "response")
prob_1se2 <- predict(ridge_logit_1se2, newx = X_test2, type = "response")

pred_min2 <- ifelse(prob_min2 > 0.5, 1, 0)
pred_1se2 <- ifelse(prob_1se2 > 0.5, 1, 0)

# Accuracy and confusion matrices
acc_min2 <- mean(pred_min2 == y_test2)
acc_1se2 <- mean(pred_1se2 == y_test2)

cat("Accuracy (lambda.min, new split):", acc_min2, "\n")
cat("Accuracy (lambda.1se, new split):", acc_1se2, "\n")

acc_min2 <- mean(pred_min2 == y_test2)
acc_1se2 <- mean(pred_1se2 == y_test2)
acc_min2
acc_1se2
## prediction accuracy is similar between both data splits

## Task 14: INTERPRETATION --> MICHAEL AND FRANCESCO

##... which model is the best? 
## If you could acquire more data, would you prefer having more variables (columns), 
## or more countries available (more rows). Why?: ....
