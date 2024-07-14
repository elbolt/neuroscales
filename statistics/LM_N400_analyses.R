# --------------------------------
# Load libraries and data
# --------------------------------
library(rstudioapi)
library(yaml)
library(tidyr)
library(dplyr)
library(ggplot2)

# Set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load configuration
config <- read_yaml("statistics_config.yaml")
data_filepath <- file.path(config$data_folder, config$N400_filename)
data <- read.csv(data_filepath)

# Preprocess data
data$participant_id <- as.factor(data$participant_id)
data$condition <- as.factor(data$condition)
data$condition <- relevel(data$condition, ref = "context")
data$MoCA_group <- as.factor(data$MoCA_group)
data$MoCA_group <- relevel(data$MoCA_group, ref = "normal")
data$PTA_z <- scale(data$PTA_dB)
data$age_z <- scale(data$age)
data$MoCA_z <- scale(data$MoCA_score)

# --------------------------------
# N400 descriptives
# --------------------------------
mean_N400 <- mean(data$N400_latency_ms)
sd_N400 <- sd(data$N400_latency_ms)
print(paste("Latency, M =", round(mean_N400, 0), "SD =", round(sd_N400, 0)))

# --------------------------------
# Condition comparison
# --------------------------------
random_condition <- data[data$condition == "random", "amplitude_uV"]
context_condition <- data[data$condition == "context", "amplitude_uV"]
t_test_result <- t.test(random_condition, context_condition, paired = TRUE)
print(t_test_result)

# --------------------------------
# Linear group model
# --------------------------------
data_wide <- data %>%
  pivot_wider(names_from = condition, values_from = amplitude_uV) %>%
  mutate(amplitude_diff_uV = context - random)

model <- lm(amplitude_diff_uV ~ MoCA_group * PTA_z + age_z, data = data_wide)

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

write.csv(model_table, file = "N400_model_table.csv", row.names = TRUE)