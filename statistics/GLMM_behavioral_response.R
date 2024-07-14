# --------------------------------
# Load libraries and data
# --------------------------------
library(rstudioapi)
library(yaml)
library(dplyr)
library(readr)
library(lme4)
library(emmeans)
library(ggplot2)
library(xtable)

# Set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load configuration
config <- read_yaml("statistics_config.yaml")
data_folder <- config$data_folder
csv_file <- config$track_files_parameters$csv_filename
factor_cols <- config$R_parameters$factor_columns
integer_cols <- config$R_parameters$integer_columns

# Load and preprocess data
data <- read_csv(file.path(data_folder, csv_file))
for (factor in factor_cols) {
  data[[factor]] <- as.factor(data[[factor]])
}
data$MoCA_group <- relevel(data$MoCA_group, ref = "normal")
data$correct <- relevel(data$correct, ref = "0")

# Select and deduplicate data
behavioral_data <- data %>%
  select(
    participant_id,
    MoCA_group,
    MoCA_score,
    age,
    PTA_dB,
    stimulus_id,
    condition,
    correct,
    RT
  ) %>%
  distinct()

# Standardize columns
behavioral_data$age_z <- scale(behavioral_data$age)
behavioral_data$MoCA_z <- scale(behavioral_data$MoCA_score)
behavioral_data$PTA_z <- scale(behavioral_data$PTA_dB)

# --------------------------------
# Model structure
# --------------------------------
final_model <- glmer(
  correct ~ MoCA_group * PTA_z + condition + age_z +
    (1 | participant_id) +
    (1 | stimulus_id),
  data = behavioral_data,
  family = binomial,
  control = glmerControl(optimizer = "nlminbwrap")
)
summary(final_model)

# --------------------------------
# Check model assumptions
# --------------------------------
# Step 1: Residuals analysis
residuals_data <- data.frame(
  residuals = residuals(final_model, type = "pearson")
)

ggplot(residuals_data, aes(x = residuals)) +
  geom_histogram(bins = 30) +
  ggtitle("Histogram of pearson residuals")

residuals_data$fitted <- predict(final_model, type = "response")

ggplot(residuals_data, aes(x = fitted, y = residuals)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "loess") +
  ggtitle("Residuals vs. fitted values")

# Step 2: Overdispersion
observed_deviance <- sum(residuals(final_model, type = "deviance")^2)
df_residuals <- df.residual(final_model)
overdispersion_factor <- observed_deviance / df_residuals
print(paste("Overdispersion factor:", overdispersion_factor))

# Step 3: Random effects analysis
ran_ef <- ranef(final_model, condVar = TRUE)
ggplot(ran_ef$participant_id, aes(x = `(Intercept)`)) +
  geom_histogram(bins = 30) +
  ggtitle("Histogram of random effects for participant ID")

# --------------------------------
# Summarize model results
# --------------------------------
# Step 1: Calculate Odds Ratios and Confidence Intervals
fixed_effects <- fixef(final_model)
odds_ratios <- exp(fixed_effects)
conf_int <- confint(final_model, method = "Wald", nsim = 5000)
odds_ratio_conf_int <- exp(conf_int)

# Step 2: Create a table with the results
results <- data.frame(
  Predictors = c(
    "Intercept",
    "MoCA group (low)",
    "PTA ($z$)",
    "Age ($z$)",
    "Condition (random)",
    "MoCA group * PTA"
  ),
  OR = sprintf("%.2f", odds_ratios),
  LL = sprintf("%.2f", odds_ratio_conf_int[3:8, 1]),
  UL = sprintf("%.2f", odds_ratio_conf_int[3:8, 2]),
  z = sprintf("%.3f", summary(final_model)$coefficients[, 3]),
  p = sprintf("%.3f", summary(final_model)$coefficients[, 4]),
  sign. = ifelse(
    summary(final_model)$coefficients[, 4] < 0.001, "***",
    ifelse(summary(final_model)$coefficients[, 4] < 0.01, "**",
           ifelse(summary(final_model)$coefficients[, 4] < 0.05, "*", ""))
  )
)

# Step 3: Convert results dataframe to a LaTeX
latex_table <- xtable(
  results, caption = "Summary of GLMM analysis", label = "tab:modelsummary",
  align = c("l", "l", "r", "r", "r", "r", "r", "l")
)
print(
  latex_table,
  type = "latex",
  include.rownames = FALSE,
  floating = FALSE,
  hline.after = c(-1, 0, nrow(results))
)

# --------------------------------
# Post-hoc analysis
# --------------------------------

# Effect plot for MoCA_group
emm_MoCA <- emmeans(final_model, ~ MoCA_group)
emm_MoCA_df <- as.data.frame(emm_MoCA)

ggplot(emm_MoCA_df, aes(x = MoCA_group, y = emmean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), width = 0.2) +
  labs(title = "Estimated marginal means for MoCA group",
       x = "MoCA group",
       y = "Estimated marginal mean (probability of correct response)") +
  theme_minimal()

# Effect plot for condition
emm_condition <- emmeans(final_model, ~ condition)
emm_condition_df <- as.data.frame(emm_condition)

ggplot(emm_condition_df, aes(x = condition, y = emmean)) +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = asymp.LCL, ymax = asymp.UCL), width = 0.2) +
  labs(title = "Estimated marginal means for condition",
       x = "Condition",
       y = "Estimated marginal mean (probability of correct response)") +
  theme_minimal()