# --------------------------------
# Load libraries and data
# --------------------------------
library(rstudioapi)
library(yaml)
library(readr)
library(lme4)
library(lmerTest)
library(emmeans)
library(ggplot2)

# Set working directory to script location
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Load configuration
config <- read_yaml("statistics_config.yaml")
data_folder <- config$data_folder
csv_file <- config$track_files_parameters$csv_filename
factor_cols <- config$R_parameters$factor_columns
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

# ------------------------------
# Analyze word-based interaction
# ------------------------------
rate <- rates_list[2]
subset_data <- data[data$frequency_band == rate, ]

model <- lmer(
  tracking_value ~ MoCA_group * PTA_z * condition + cluster + age_z +
    (1 | stimulus_id) +
    (1 | participant_id),
  data = subset_data,
  control = lmerControl(optimizer = "bobyqa")
)

summary(model)

# Post-hoc analysis
emm <- emmeans(model, ~ MoCA_group * condition)
pairs(emm, by = "condition")

# Plot
emm_df <- as.data.frame(emm)
ggplot(emm_df, aes(x = condition, y = emmean, color = MoCA_group, group = MoCA_group)) +
  geom_point(position = position_dodge(width = 0.3), size = 3) +
  geom_line(position = position_dodge(width = 0.3)) +
  geom_errorbar(aes(ymin = emmean - SE, ymax = emmean + SE), 
                width = 0.2, position = position_dodge(width = 0.3)) +
  labs(x = "Condition", y = "Estimated marginal means of word rate tracking",
       color = "MoCA group") +
  theme_minimal() +
  ggtitle("Interaction of MoCA group and condition on word rate tracking") +
  theme(plot.title = element_text(hjust = 0.5))

# -----------------------------------
# Analyze syllable-based interactions
# -----------------------------------
rate <- rates_list[3]
subset_data <- data[data$frequency_band == rate, ]

model <- lmer(
  tracking_value ~ MoCA_group * PTA_z * condition + cluster + age_z +
    (1 | stimulus_id) +
    (1 | participant_id),
  data = subset_data,
  control = lmerControl(optimizer = "bobyqa")
)

summary(model)

# Post-hoc analysis on MoCA and condition
emm <- emmeans(model, ~ MoCA_group * condition)
pairs(emm, by = "condition")

# Post-hoc analysis on PTA and condition
mean_PTA_z <- mean(data$PTA_z)
sd_PTA_z <- sd(data$PTA_z)
PTA_z_values <- c(mean_PTA_z - sd_PTA_z, mean_PTA_z, mean_PTA_z + sd_PTA_z)

emm <- emmeans(model, ~ PTA_z * condition, at = list(PTA_z = PTA_z_values))
pairs(emm, by = "condition")

# Extract the estimated marginal means for plotting
emm_df <- as.data.frame(emm)

# Create the interaction plot
ggplot(emm_df, aes(x = PTA_z, y = emmean, color = condition, group = condition)) +
  geom_line() +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = emmean - SE, ymax = emmean + SE), width = 0.1) +
  labs(x = "PTA (z-scored)", y = "Estimated tracking value", color = "Condition") +
  theme_minimal() +
  ggtitle("Interaction between PTA and condition on syllable rate tracking")

# -----------------------------------
# Analyze phoneme-based interactions
# -----------------------------------
rate <- rates_list[4]
subset_data <- data[data$frequency_band == rate, ]

model <- lmer(
  tracking_value ~ MoCA_group * PTA_z * condition + cluster + age_z +
    (1 | stimulus_id) +
    (1 | participant_id),
  data = subset_data,
  control = lmerControl(optimizer = "bobyqa")
)

summary(model)

# Post-hoc analysis on MoCA and condition
emm <- emmeans(model, ~ MoCA_group * condition)
pairs(emm, by = "condition")

# Plot
emm_df <- as.data.frame(emm)
ggplot(emm_df, aes(x = condition, y = emmean, color = MoCA_group, group = MoCA_group)) +
  geom_point(position = position_dodge(width = 0.3), size = 3) +
  geom_line(position = position_dodge(width = 0.3)) +
  geom_errorbar(aes(ymin = emmean - SE, ymax = emmean + SE), 
                width = 0.2, position = position_dodge(width = 0.3)) +
  labs(x = "Condition", y = "Estimated marginal means of phoneme rate tracking",
       color = "MoCA group") +
  theme_minimal() +
  ggtitle("Interaction of MoCA group and condition on phoneme rate tracking") +
  theme(plot.title = element_text(hjust = 0.5))

# Post-hoc analysis on PTA and condition
mean_PTA_z <- mean(data$PTA_z)
sd_PTA_z <- sd(data$PTA_z)
PTA_z_values <- c(mean_PTA_z - sd_PTA_z, mean_PTA_z, mean_PTA_z + sd_PTA_z)

emm <- emmeans(model, ~ PTA_z * condition, at = list(PTA_z = PTA_z_values))
pairs(emm, by = "condition")

# Extract the estimated marginal means for plotting
emm_df <- as.data.frame(emm)

# Create the interaction plot
ggplot(emm_df, aes(x = PTA_z, y = emmean, color = condition, group = condition)) +
  geom_line() +
  geom_point(size = 3) +
  geom_errorbar(aes(ymin = emmean - SE, ymax = emmean + SE), width = 0.1) +
  labs(x = "PTA (z-scored)", y = "Estimated tracking value", color = "Condition") +
  theme_minimal() +
  ggtitle("Interaction between PTA and condition on phoneme rate tracking")