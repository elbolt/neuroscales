# --------------------------------
# Load libraries and data
# --------------------------------
library(rstudioapi)
library(yaml)
library(readr)
library(lme4)
library(lmerTest)

# Set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load configuration
config <- read_yaml("statistics_config.yaml")
data_folder <- config$data_folder
csv_file <- config$track_files_parameters$csv_filename
factor_cols <- config$R_parameters$factor_columns
integer_cols <- config$R_parameters$integer_columns
rates_list <- config$R_parameters$rates_list

# Load and preprocess data
data <- read_csv(file.path(data_folder, csv_file))
for (factor in factor_cols) {
  data[[factor]] <- as.factor(data[[factor]])
}
data$MoCA_group <- relevel(data$MoCA_group, ref = "normal")
data$condition <- relevel(data$condition, ref = "context")
data$cluster <- relevel(data$cluster, ref = "F")
data <- data[data$correct == 1, ]

# Standardize columns
data$age_z <- scale(data$age)
data$MoCA_z <- scale(data$MoCA_score)
data$PTA_z <- scale(data$PTA_dB)

# ----------------------------
# Analyze rate-based models
# ----------------------------
for (rate in rates_list) {
  subset_data <- data[data$frequency_band == rate, ]
  
  model <- lmer(
    tracking_value ~ MoCA_group * PTA_z * condition + cluster + age_z +
      (1 | stimulus_id) +
      (1 | participant_id),
    data = subset_data,
    control = lmerControl(optimizer = "bobyqa")
  )
  
  ci_table <- confint(
    model,
    parm = "beta_",
    method = "Wald",
    nsim = 5000,
    seed = 2025
  )
  
  # --------------------------------
  # Model tables
  # --------------------------------
  model_table <- as.data.frame(summary(model)$coefficients)
  
  model_table$LL <- ci_table[, 1]
  model_table$UL <- ci_table[, 2]
  
  model_table <- model_table[, c(1, 6, 7, 3, 4, 5)]
  model_table <- round(model_table, 3)
  model_table$df <- round(model_table$df, 1)
  model_table$`t value` <- round(model_table$`t value`, 1)
  
  p_values <- model_table[, "Pr(>|t|)"]
  sign_list <- c()
  for (p in p_values) {
    if (p < 0.001) {
      sign_list <- c(sign_list, "***")
    } else if (p < 0.01) {
      sign_list <- c(sign_list, "**")
    } else if (p < 0.05) {
      sign_list <- c(sign_list, "*")
    } else {
      sign_list <- c(sign_list, "")
    }
  }
  model_table$sign. <- sign_list
  
  model_table$Estimate <- paste0("$", sprintf("%.3f", model_table$Estimate), "$")
  model_table$LL <- paste0("$", sprintf("%.3f", model_table$LL), "$")
  model_table$UL <- paste0("$", sprintf("%.3f", model_table$UL), "$")
  model_table$`t value` <- paste0("$", sprintf("%.1f", model_table$`t value`), "$")
  model_table$df <- paste0("$", model_table$df, "$")
  model_table$`Pr(>|t|)` <- paste0("$", sprintf("%.3f", model_table$`Pr(>|t|)`), "$")
  
  rownames <- c(
    "Intercept",
    "MoCA group (low)",
    "PTA ($z$)",
    "Condition (random)",
    "Cluster (C)",
    "Cluster (P)",
    "Age ($z$)",
    "MoCA group $\times$ PTA",
    "MoCA group $\times$ Condition",
    "PTA $\times$ Condition",
    "MoCA group $\times$ PTA $\times$ Condition"
  )
  rownames(model_table) <- rownames
  
  headernames <- c("Estimate", "LL", "UL", "$df$", "$t$", "$p$", "sign.")
  colnames(model_table) <- headernames
  
  model_table <- apply(model_table, 2, function(x) {
    if (is.numeric(x)) {
      x <- paste0("$", x, "$")
    }
    return(x)
  })
  
  write.csv(model_table, paste("tracking_model_table_", rate, ".csv", sep = ""))
}