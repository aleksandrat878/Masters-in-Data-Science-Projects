library(randomForest)
library(iml)
library(ggplot2)
library(tidyverse)
library(dplyr)
library(reshape2) 
library(patchwork)
library(kableExtra)
library(knitr)


load("TransportModeSweden.2526.RData")

#-------------- Preprocessing---------------
str(data)
colSums(is.na(data))
data$walk_500[is.na(data$walk_500)] <- 0

data$mode <- as.factor(data$mode)

table(data$mode)

# Fix decimal commas and convert to numeric
numeric_vars <- c("time_pt", "time_car", "time_ratio")

data[numeric_vars] <- lapply(data[numeric_vars], function(x) {
  x <- gsub(",", ".", x)     # replace comma with dot
  as.numeric(x)
})
# Random forests do NOT require scaling, normalization, distribution checks, etc.

#-------------- Model-------------
set.seed(123)

test_idx <- sample(1:nrow(data), 10)  
test <- data[test_idx, ]
train <- data[-test_idx, ]

# Train Random Forest Model
rf <- randomForest(
  mode ~ ., 
  data = train,
  ntree = 500,
  importance = TRUE
)

#check m

print(rf)

# Variable importance plot (Appendix A)
imp <- as.data.frame(importance(rf))
imp$Feature <- rownames(imp)

ggplot(imp, aes(x = reorder(Feature, MeanDecreaseGini),
                y = MeanDecreaseGini)) +
  geom_col(fill = "#4C72B0") +
  coord_flip() +
  labs(title = "Random Forest Variable Importance",
       x = "Feature", y = "Mean Decrease Gini") +
  theme_minimal()



# Predictions on Test Set
pred <- predict(rf, test)
comparison <- data.frame( actual = test$mode, predicted = pred)

# Identify one correct + one incorrect case
correct_idx   <- which(comparison$actual == comparison$predicted)[1]
incorrect_idx <- which(comparison$actual != comparison$predicted)[1]

correct_case   <- test[correct_idx, ]
incorrect_case <- test[incorrect_idx, ]

confusionMatrix <- table(predicted = pred, actual = test$mode)
confusionMatrix


# ------ SHAP Explanations ---------
predictor <- Predictor$new( model = rf, data = train %>% 
                              select(-mode),  
                              y = train$mode, type = "prob")

# Check prediction for person A and person B
predictor$predict(correct_case %>% select(-mode))
predictor$predict(incorrect_case %>% select(-mode))


# SHAP for correct
set.seed(100)
pred_class_A <- as.numeric(as.character(correct_case$mode))

shap_correct <- Shapley$new(
  predictor,
  x.interest = correct_case %>% select(-mode)
)

# filter for predicted class only
df_correct_filtered <- shap_correct$results %>%
  filter(class == pred_class_A) %>%
  mutate(direction = ifelse(phi > 0, "Positive", "Negative"))


P1 <- ggplot(df_correct_filtered,
       aes(x = phi,
           y = reorder(feature, phi),
           fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c(
    "Positive" = "#1f78b4",   # blue
    "Negative" = "#e31a1c"    # red
  )) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal(base_size = 14) +
  labs(
    title = paste0("SHAP – Correct Prediction (Case: 2463)\nActual prediction: ",
                   round(predictor$predict(correct_case %>% select(-mode))[pred_class_A+1], 2)),
    x = "SHAP value",
    y = "Feature",
    fill = "Direction"
  )

# SHAP for incorrect
set.seed(100)
pred_class_B <- as.numeric(as.character(incorrect_case$mode))

shap_incorrect <- Shapley$new(
  predictor,
  x.interest = incorrect_case %>% select(-mode)
)

# filter for predicted class only
df_incorrect_filtered <- shap_incorrect$results %>%
  filter(class == pred_class_B) %>%
  mutate(direction = ifelse(phi > 0, "Positive", "Negative"))

P2 <- ggplot(df_incorrect_filtered,
             aes(x = phi,
                 y = reorder(feature, phi),
                 fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c(
    "Positive" = "#1f78b4",   # blue
    "Negative" = "#e31a1c"    # red
  )) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal(base_size = 14) +
  labs(
    title = paste0("SHAP – Incorrect Prediction (Case 1038)\nActual prediction: ",
                   round(predictor$predict(incorrect_case %>% select(-mode))[pred_class_B+1], 2)),
    x = "SHAP value",
    y = "Feature",
    fill = "Direction"
  )

P1 + P2


# ---------- LIME Interpretation ------------
predictor <- Predictor$new(
  model = rf,
  data = train %>% select(-mode),
  y = train$mode,
  type = "prob",
  class = "1"   # EXPLAIN PROBABILITY OF CLASS 1 ONLY
)

set.seed(100)

# LIME correct prediction
set.seed(100)

lime_correct <- LocalModel$new(
  predictor,
  x.interest = correct_case %>% select(-mode),
  k = 5   # number of features to display in the explanation
)

df_lime_correct <- lime_correct$results %>%
  mutate(direction = ifelse(effect > 0, "Positive", "Negative"))

p_lime_correct <- ggplot(df_lime_correct,
                         aes(x = effect,
                             y = reorder(feature, effect),
                             fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c(
    "Positive" = "#1f78b4",   # blue (same as SHAP)
    "Negative" = "#e31a1c"    # red (same as SHAP)
  )) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none") +
  labs(
    title = "LIME – Correct Prediction (Case 2463)",
    x = "Local Contribution (LIME Effect)",
    y = "Feature"
  )

# ---------------- LIME FOR INCORRECT PREDICTION ----------------

lime_incorrect <- LocalModel$new(
  predictor,
  x.interest = incorrect_case %>% select(-mode),
  k = 5   # number of features to display in the explanation
)

df_lime_incorrect <- lime_incorrect$results %>%
  mutate(direction = ifelse(effect > 0, "Positive", "Negative"))

p_lime_incorrect <- ggplot(df_lime_incorrect,
                           aes(x = effect,
                               y = reorder(feature, effect),
                               fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c(
    "Positive" = "#1f78b4",
    "Negative" = "#e31a1c"
  )) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none") +
  labs(
    title = "LIME – Incorrect Prediction (Case 1038)",
    x = "Local Contribution (LIME Effect)",
    y = "Feature"
  )

p_lime_correct + p_lime_incorrect












lime_correct
df_lime_correct <- lime_correct$results %>% 
  mutate(direction = ifelse(effect > 0, "Positive", "Negative"))
p_lime_correct <-  lime_correct$results %>% 
  mutate(direction = ifelse(effect > 0, "Positive", "Negative"))

p_lime_correct <- ggplot(df_lime_correct, 
                         aes(x = effect, 
                             y = reorder(feature, effect),
                             fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c("Positive" = "#4C72B0",   # blue
                               "Negative" = "#D55E00")) + # red
  theme_minimal(base_size = 12) +
  theme(legend.position = "none") +   # ← REMOVE LEGEND
  labs(title = "LIME – Correct Prediction (Person A)",
       x = "Local Contribution (Effect)",
       y = "Feature")
# LIME incorrect prediction
set.seed(123)

lime_incorrect <- LocalModel$new(
  predictor,
  x.interest = incorrect_case %>% select(-mode),
  k = 5
)

lime_incorrect
df_lime_incorrect <- lime_incorrect$results %>% 
  mutate(direction = ifelse(effect > 0, "Positive", "Negative"))

p_lime_incorrect <- ggplot(df_lime_incorrect, 
                           aes(x = effect, 
                               y = reorder(feature, effect),
                               fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c("Positive" = "#4C72B0",
                               "Negative" = "#D55E00"),
                    labels = c("Positive" = "Supports Predicted Class",
                               "Negative" = "Contradicts Predicted Class")) +
  theme_minimal(base_size = 12) +
  labs(title = "LIME – Incorrect Prediction (Person B)",
       x = "Local Contribution (Effect)",
       y = "Feature")


(p_lime_correct + p_lime_incorrect)


lime_correct$results
lime_incorrect$results

correct_case
incorrect_case

# Add row names as an ID column
correct_tbl <- correct_case %>%
  mutate(ID = rownames(correct_case)) %>%
  select(ID, everything())

incorrect_tbl <- incorrect_case %>%
  mutate(ID = rownames(incorrect_case)) %>%
  select(ID, everything())

# Kable tables
kable(correct_tbl,
      caption = "Profile of Correctly Classified Individual (Person A)",
      digits = 3) %>%
  kable_styling(full_width = FALSE)

kable(incorrect_tbl,
      caption = "Profile of Incorrectly Classified Individual (Person B)",
      digits = 3) %>%
  kable_styling(full_width = FALSE)




### ---------------- DISPLAY BOTH PLOTS ----------------
plot_lime_correct
plot_lime_incorrect


# -----------


P1 <- ggplot(df_correct_filtered,
             aes(x = phi,
                 y = reorder(feature, phi),
                 fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c("Positive" = "#1f78b4",
                               "Negative" = "#e31a1c")) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "none",
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    plot.title = element_text(size = 10),
    plot.margin = margin(5,5,5,5)
  ) +
  labs(
    title = paste0("SHAP – Correct Prediction (Case 2463)\nPrediction: ",
                   round(predictor$predict(correct_case %>% select(-mode))[pred_class_A+1], 2))
  )


P1 <- ggplot(df_correct_filtered,
             aes(x = phi,
                 y = reorder(feature, phi),
                 fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c("Positive" = "#1f78b4",
                               "Negative" = "#e31a1c")) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "none",
    axis.title = element_blank(),
    axis.text = element_text(size = 8),
    plot.title = element_text(size = 10),
    plot.margin = margin(5,5,5,5)
  ) +
  labs(title = "SHAP – Correct Prediction (Case 2463)")



P1 <- ggplot(df_correct_filtered,
             aes(x = phi,
                 y = reorder(feature, phi),
                 fill = direction)) +
  geom_col() +
  scale_fill_manual(values = c("Positive" = "#1f78b4",
                               "Negative" = "#e31a1c")) +
  geom_vline(xintercept = 0, linetype = "dashed") +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "none",
    axis.title.y = element_blank(),     # REMOVE Y AXIS TITLE
    axis.title.x = element_text(size = 9),  # KEEP X AXIS TITLE
    axis.text = element_text(size = 8),
    plot.title = element_text(size = 10),
    plot.margin = margin(5,5,5,5)
  ) +
  labs(
    title = "SHAP – Correct Prediction (Case 2463)",
    x = "SHAP value"   # KEEP THIS
  )
