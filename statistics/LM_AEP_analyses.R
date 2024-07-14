# --------------------------------
# Load libraries and data
# --------------------------------
library(rstudioapi)
library(yaml)
library(dplyr)
library(ggplot2)

# Set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load configuration
config <- read_yaml("statistics_config.yaml")
data_filepath <- file.path(config$data_folder, config$AEP_filename)
data <- read.csv(data_filepath)

# Preprocess data
data$participant_id <- as.factor(data$participant_id)
data$MoCA_group <- as.factor(data$MoCA_group)
data$MoCA_group <- relevel(data$MoCA_group, ref = "normal")
data$PTA_z <- scale(data$PTA_dB)
data$age_z <- scale(data$age)
data$MoCA_z <- scale(data$MoCA_score)

components <- c("P1", "N1", "P2")

for (component in components) {
  data <- data %>%
    mutate(
      amplitude_uV = case_when(
        component == "P1" ~ P1_amplitude_uV,
        component == "N1" ~ N1_amplitude_uV,
        component == "P2" ~ P2_amplitude_uV
      ),
      latency_ms = case_when(
        component == "P1" ~ P1_latency_ms,
        component == "N1" ~ N1_latency_ms,
        component == "P2" ~ P2_latency_ms
      )
    )
  
  # --------------------------------
  # Descriptives
  # --------------------------------
  mean_peak <- mean(data$latency_ms)
  sd_peak <- sd(data$latency_ms)
  ttest <- t.test(latency_ms ~ MoCA_group, data = data, tail = "two.sided")
  eta_squared <- ttest$statistic^2 / (ttest$statistic^2 + ttest$parameter)
  print(paste("Component:", component, "Mean latency =", round(mean_peak, 0), "SD =", round(sd_peak, 0), "Eta squared =", round(eta_squared, 3)))
  
  # --------------------------------
  # Linear group model
  # --------------------------------
  model <- lm(amplitude_uV ~ MoCA_group * PTA_z + age_z, data = data)
  
  # --------------------------------
  # Model tables
  # --------------------------------
  model_table <- as.data.frame(summary(model)$coefficients)
  ci_table <- confint(model)
  model_table$LL <- ci_table[, 1]
  model_table$UL <- ci_table[, 2]
  
  model_table <- model_table[, c(1, 5, 6, 3, 4)]
  model_table <- round(model_table, 3)
  model_table$`t value` <- round(model_table$`t value`, 1)
  
  p_values <- model_table[, "Pr(>|t|)"]
  sign_list <- sapply(p_values, function(p) {
    if (p < 0.001) "***" else if (p < 0.01) "**" else if (p < 0.05) "*" else ""
  })
  model_table$sign. <- sign_list
  
  model_table$Estimate <- paste0("$", sprintf("%.3f", model_table$Estimate), "$")
  model_table$LL <- paste0("$", sprintf("%.3f", model_table$LL), "$")
  model_table$UL <- paste0("$", sprintf("%.3f", model_table$UL), "$")
  model_table$`t value` <- paste0("$", sprintf("%.1f", model_table$`t value`), "$")
  model_table$`Pr(>|t|)` <- paste0("$", sprintf("%.3f", model_table$`Pr(>|t|)`), "$")
  
  rownames <- c(
    "Intercept",
    "MoCA group (low)",
    "PTA ($z$)",
    "Age ($z$)",
    "MoCA group (low) * PTA ($z$)"
  )
  rownames(model_table) <- rownames
  
  headernames <- c("Estimate", "LL", "UL", "$t$", "$p$", "sign.")
  colnames(model_table) <- headernames
  
  write.csv(model_table,
            file = paste0("AEP_model_table_", component, ".csv"),
            row.names = TRUE
  )
}