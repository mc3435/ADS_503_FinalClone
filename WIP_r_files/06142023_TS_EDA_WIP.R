library(ggplot2)
library(dplyr)
library(tidyr)
library(mlbench)
library(scales)
library(patchwork)
library(gridExtra)
library(reshape2)
library(GGally)
library(cowplot)
library(moments)
library(MASS)
library(readr)
library(e1071)
library(car)
library(tidyverse)
library(devtools)
library(caret)
library(fortunes)
library(recipes)
library(httpuv)
library(pkgbuild)
library(htmlwidgets)
library(sessioninfo)
library(glmnet)
library(randomForest)
library(AppliedPredictiveModeling)
library(pls)
library(magrittr)
library(knitr)
library(elasticnet)
library(RColorBrewer)
library(chemometrics)
library(rpart)
library(ggpubr)
library(mice)
library(RANN)
library(kernlab)
library(mlbench)
library(randomForest)
library(scales)
library(MASS)
library(pamr)
library(cluster)
library(survival)
library(pROC)
library(PerformanceAnalytics)
library(psych)
library(scales)
library(knitr)
library(reshape2)
library(corrplot)
library(relaimpo)
library(factoextra)
library(Rtsne)
library(xgboost)
library(ROSE)

install.packages("ROSE")

# Load the CSV file into a dataframe & assess the data
raw.data <- read.csv('C:/MIDS/ADS-503_Applied_Predictive_Modeling/ADS_503_team_2_final_project/data/breast_cancer_FNA_data.csv')
df <- read.csv('C:/MIDS/ADS-503_Applied_Predictive_Modeling/ADS_503_team_2_final_project/data/breast_cancer_FNA_data.csv', header=T)
print(sprintf("Number of data columns: %d",ncol(df)))
df
variable_names <- colnames(df)
print(variable_names)
knitr::kable(head(df,5),caption="Raw data (first 5 rows)")
glimpse(df)
summary(df)
diagnostic <- plyr::count(df$diagnosis)
print(sprintf("Malignant: %d | Benign: %d",diagnostic$freq[2],diagnostic$freq[1]))
print(sprintf("Percent of malignant tumor: %.1f%%",round(diagnostic$freq[2]/nrow(df)*100,1)))

# Create separate data frames for each metric for each of the features

df_clean <- df[, !(names(df) %in% c("X", "id"))]
df_clean
variable_names <- colnames(df_clean)
print(variable_names)

# Create empty data frames
df_mean <- data.frame(diagnosis = df_clean$diagnosis)
df_se <- data.frame(diagnosis = df_clean$diagnosis)
df_worst <- data.frame(diagnosis = df_clean$diagnosis)

# Loop over variable names
for (variable_name in variable_names) {
  # Extract the part after the "_" delimiter
  variable_type <- sub(".+_([^_]+)$", "\\1", variable_name)
  # Assign the column to the appropriate data frame based on variable type
  if (variable_type == "mean") {
    df_mean[[variable_name]] <- df_clean[[variable_name]]
  } else if (variable_type == "se") {
    df_se[[variable_name]] <- df_clean[[variable_name]]
  } else if (variable_type == "worst") {
    df_worst[[variable_name]] <- df_clean[[variable_name]]
  }
}

# Show feature distributions for the target variable 'diagnosis' in each data frame
scales <- list(x = list(relation = "free"), y = list(relation = "free"), cex = 0.6)
print(featurePlot(x = df_mean, y = df_mean$diagnosis, plot = "density", scales = scales,
            layout = c(3, 10), auto.key = list(columns = 2), pch = "|"))
featurePlot(x = df_se, y = df_se$diagnosis, plot = "density", scales = scales,
            layout = c(3, 10), auto.key = list(columns = 2), pch = "|")
featurePlot(x = df_worst, y = df_worst$diagnosis, plot = "density", scales = scales,
            layout = c(3, 10), auto.key = list(columns = 2), pch = "|")

### Analyze the Feature Density & Correlation between variables
# Visualization for the "Mean" Variables
ggpairs(df_mean, aes(color=diagnosis, alpha=0.75), lower=list(continuous="smooth"))+ theme_bw()+
  labs(title="Feature Density & Correlation - Cancer Means")+theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=20))
# Visualization for the "Standard Error" Variables
ggpairs(df_se, aes(color=diagnosis, alpha=0.75), lower=list(continuous="smooth"))+ theme_bw()+
  labs(title="Feature Density & Correlation - Cancer Standard Error")+theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=20))
# Visualization for the "Worst" Variables
ggpairs(df_worst, aes(color=diagnosis, alpha=0.75), lower=list(continuous="smooth"))+ theme_bw()+
  labs(title="Feature Density & Correlation - Cancer Worst (Mean of the three largest values)")+theme(plot.title=element_text(face='bold',color='black',hjust=0.5,size=20))

# Pearson Correlation of all features
df_clean$diagnosis <- as.integer(factor(df_clean$diagnosis))-1
df_clean
correlations <- cor(df_clean,method="pearson")
corrplot(correlations, number.cex = .9, method = "square", 
         hclust.method = "ward", order = "FPC",
         type = "full", tl.cex=0.8,tl.col = "black")

### Calculate the VIF values (using linear model) for the predictor variables to look for multicollinearity 
lm_model <- lm(diagnosis ~ ., data = df_clean)
vif_values <- vif(lm_model)
vif_table <- data.frame(Variable = names(vif_values), VIF = vif_values)
vif_table <- vif_table[order(vif_table$VIF), ]
vif_table$VIF <- round(vif_table$VIF)
vif_plot <- ggplot(vif_table, aes(x = reorder(Variable, VIF), y = VIF)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = VIF), vjust = -0.5) +  # Add integer value labels
  labs(x = "Variable", y = "Variance Inflation Factor (VIF)") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("VIF Values of Predictor Variables (Ascending Order)")
print(vif_plot)

### Calculate the eigenvalues for the predictor variables to look for multicollinearity 
# Small eigenvalues suggest potential multicollinearity issues

correlations <- cor(df_clean[-1], method = "pearson")
eigenvalues <- eigen(correlations)$values
eigen_table <- data.frame(Variable = colnames(correlations), Eigenvalue = eigenvalues)
eigen_table <- eigen_table[eigen_table$Variable != "diagnosis", ]
eigen_table <- eigen_table[order(eigen_table$Eigenvalue), ]
eigen_plot <- ggplot(eigen_table, aes(x = reorder(Variable, Eigenvalue), y = Eigenvalue)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Variable", y = "Eigenvalue") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Eigenvalues of Variables (Ascending Order)")
eigen_plot

### Calculate the tolerance (reciprocal of VIF) for the predictor variables to look for multicollinearity 
tolerance <- 1 / vif_values
tolerance_table <- data.frame(Variable = names(tolerance), Tolerance = tolerance)
tolerance_table <- tolerance_table[order(tolerance_table$Tolerance), ]
tolerance_plot <- ggplot(tolerance_table, aes(x = reorder(Variable, Tolerance), y = Tolerance)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  labs(x = "Variable", y = "Tolerance") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Tolerance Values of Predictor Variables (Ascending Order)")
tolerance_plot

### Principal Components Analysis (PCA) transform

# Perform PCA
var_only <- df_clean[, !(names(df_clean) %in% c("diagnosis"))] # predictor variables only
diagnosis <- raw.data[, 2] # target variable only 
var_pca <- prcomp(var_only, center = TRUE, scale. = TRUE)

# Plot PCA with customized aesthetics
fviz_eig(var_pca, addlabels = TRUE, ylim = c(0, 50),
         title = "Principal Components Analysis (PCA)",
         subtitle = NULL,
         xlab = "Principal Component",
         ylab = "Proportion of Variance Explained",
         geom = "line",
         linecolor = "red",
         pointsize = 2,
         pointshape = 21,
         pointfill = "white",
         pointcolor = "red",
         legend.title = "Principal Components",
         legend.position = "right")
# Adjust plot margins
par(mar = c(5, 5, 4, 2) + 0.1)

### t-SNE transform for dimensionality reduction
colors <- c("blue", "red")
names(colors) = unique(diagnosis)
set.seed(31452)
tsne <- Rtsne(var_only, dims=2, perplexity=30, 
              verbose=TRUE, pca=TRUE, 
              theta=0.01, max_iter=1000)
plot(tsne$Y, t='n', main="t-Distributed Stochastic Neighbor Embedding (t-SNE)",
     xlab="t-SNE 1st dimm.", ylab="t-SNE 2nd dimm.")
text(tsne$Y, labels=diagnosis, cex=0.5, col=colors[diagnosis])

# Create PCA biplot with customized aesthetics
pca_biplot <- fviz_pca_biplot(var_pca,
                          geom.ind = "point",
                          col.ind = diagnosis,
                          palette = c("blue", "red"),
                          addEllipses = TRUE,
                          axes.linetype = "dashed",
                          title = "Principal Components Analysis (PCA)",
                          xlab = "PC1 (Proportion of Variance Explained: 32.3%)",
                          ylab = "PC2 (Proportion of Variance Explained: 29.7%)",
                          legend.title = "Diagnosis",
                          legend.position = "right",
                          legend.shape = "circle",
                          legend.label = c("Benign", "Malignant"))
pca_biplot

### Random Forest Modeling, Feature Importance, Tuning, and Performance Metrics

# Prepare the data for random forest
df_rf <- df_clean
df_rf$diagnosis <- as.factor(df_rf$diagnosis)

# Split the data into training and testing sets
set.seed(42)  # For reproducibility
train_indices <- sample(1:nrow(df_rf), 0.7 * nrow(df_rf))  # 70% for training
train_data <- df_rf[train_indices, ]
test_data <- df_rf[-train_indices, ]

### Original Model ###
# Train a random forest model with original parameters
rf_model_original <- randomForest(
  diagnosis ~ .,
  data = train_data,
  ntree = 500,
  importance = TRUE
)

# Predict on the test set with original model
predicted_original <- predict(rf_model_original, newdata = test_data)

# Convert factor to numeric for AUC calculation
predicted_numeric <- as.numeric(predicted_original) - 1

# Calculate performance metrics for original model
confusion_matrix_original <- table(Actual = test_data$diagnosis, Predicted = predicted_original)
accuracy_original <- sum(diag(confusion_matrix_original)) / sum(confusion_matrix_original)
precision_original <- confusion_matrix_original[2, 2] / sum(confusion_matrix_original[, 2])
recall_original <- confusion_matrix_original[2, 2] / sum(confusion_matrix_original[2, ])
f1_score_original <- 2 * (precision_original * recall_original) / (precision_original + recall_original)
specificity_original <- confusion_matrix_original[1, 1] / sum(confusion_matrix_original[1, ])
sensitivity_original <- recall_original
auc_original <- roc(test_data$diagnosis, predicted_numeric)$auc

### Recursive Feature Elimination (RFE) Model ###
# Perform recursive feature elimination
ctrl_rfe <- rfeControl(functions = rfFuncs, method = "cv", number = 5)  # 5-fold cross-validation
rfe_model <- rfe(
  x = train_data[, -ncol(train_data)],  # Exclude the target variable
  y = train_data$diagnosis,
  sizes = c(5, 10, 15),  # Different feature subset sizes to consider
  rfeControl = ctrl_rfe
)

# Get the selected features from RFE
selected_features <- predictors(rfe_model)

# Train the model using the selected features from RFE
rf_model_rfe <- randomForest(
  diagnosis ~ .,
  data = train_data[, c(selected_features, "diagnosis")],
  ntree = 500,
  importance = TRUE
)

# Predict on the test set with RFE model
test_data_rfe <- test_data[, c(selected_features, "diagnosis")]
predicted_rfe <- predict(rf_model_rfe, newdata = test_data_rfe)

# Calculate performance metrics for RFE model
confusion_matrix_rfe <- table(Actual = test_data$diagnosis, Predicted = predicted_rfe)
accuracy_rfe <- sum(diag(confusion_matrix_rfe)) / sum(confusion_matrix_rfe)
precision_rfe <- confusion_matrix_rfe[2, 2] / sum(confusion_matrix_rfe[, 2])
recall_rfe <- confusion_matrix_rfe[2, 2] / sum(confusion_matrix_rfe[2, ])
f1_score_rfe <- 2 * (precision_rfe * recall_rfe) / (precision_rfe + recall_rfe)
specificity_rfe <- confusion_matrix_rfe[1, 1] / sum(confusion_matrix_rfe[1, ])
sensitivity_rfe <- recall_rfe
predicted_rfe_numeric <- as.numeric(predicted_rfe) - 1
auc_rfe <- roc(test_data$diagnosis, predicted_rfe_numeric)$auc

### Tuned Model ###
# Define parameter grid for tuning
param_grid <- expand.grid(
  mtry = c(2, 4, 6, 8)  # Different values for mtry
)

# Perform grid search with cross-validation
ctrl_tune <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation
tuned_model <- train(
  diagnosis ~ .,
  data = train_data,
  method = "rf",
  trControl = ctrl_tune,
  tuneGrid = param_grid
)

# Get the optimal parameter values from tuning
best_mtry <- tuned_model$bestTune$mtry
best_splitrule <- tuned_model$bestTune$splitrule

# Train the model with the optimal parameters
rf_model_tuned <- randomForest(
  diagnosis ~ .,
  data = train_data,
  ntree = 500,
  mtry = best_mtry,
  splitrule = best_splitrule,
  importance = TRUE
)

# Predict on the test set with tuned model
predicted_tuned <- predict(rf_model_tuned, newdata = test_data)

# Calculate performance metrics for tuned model
confusion_matrix_tuned <- table(Actual = test_data$diagnosis, Predicted = predicted_tuned)
accuracy_tuned <- sum(diag(confusion_matrix_tuned)) / sum(confusion_matrix_tuned)
precision_tuned <- confusion_matrix_tuned[2, 2] / sum(confusion_matrix_tuned[, 2])
recall_tuned <- confusion_matrix_tuned[2, 2] / sum(confusion_matrix_tuned[2, ])
f1_score_tuned <- 2 * (precision_tuned * recall_tuned) / (precision_tuned + recall_tuned)
specificity_tuned <- confusion_matrix_tuned[1, 1] / sum(confusion_matrix_tuned[1, ])
sensitivity_tuned <- recall_tuned
predicted_tuned_numeric <- as.numeric(predicted_tuned) - 1
auc_tuned <- roc(test_data$diagnosis, predicted_tuned_numeric)$auc

### Print Performance Metrics Comparison ###
cat("Original Model:\n")
cat("Accuracy:", accuracy_original, "\n")
cat("Precision:", precision_original, "\n")
cat("Recall:", recall_original, "\n")
cat("F1-score:", f1_score_original, "\n")
cat("Specificity:", specificity_original, "\n")
cat("Sensitivity:", sensitivity_original, "\n")
cat("AUC:", auc_original, "\n\n")

cat("RFE Model:\n")
cat("Accuracy:", accuracy_rfe, "\n")
cat("Precision:", precision_rfe, "\n")
cat("Recall:", recall_rfe, "\n")
cat("F1-score:", f1_score_rfe, "\n")
cat("Specificity:", specificity_rfe, "\n")
cat("Sensitivity:", sensitivity_rfe, "\n")
cat("AUC:", auc_rfe, "\n\n")

cat("Tuned Model:\n")
cat("Accuracy:", accuracy_tuned, "\n")
cat("Precision:", precision_tuned, "\n")
cat("Recall:", recall_tuned, "\n")
cat("F1-score:", f1_score_tuned, "\n")
cat("Specificity:", specificity_tuned, "\n")
cat("Sensitivity:", sensitivity_tuned, "\n")
cat("AUC:", auc_tuned, "\n\n")

### Plot Differences in Performance Metrics ###
performance_metrics <- data.frame(
  Model = c("Original", "RFE", "Tuned"),
  Accuracy = c(accuracy_original, accuracy_rfe, accuracy_tuned),
  Precision = c(precision_original, precision_rfe, precision_tuned),
  Recall = c(recall_original, recall_rfe, recall_tuned),
  F1_score = c(f1_score_original, f1_score_rfe, f1_score_tuned),
  Specificity = c(specificity_original, specificity_rfe, specificity_tuned),
  Sensitivity = c(sensitivity_original, sensitivity_rfe, sensitivity_tuned),
  AUC = c(auc_original, auc_rfe, auc_tuned)
)

# Reshape performance metrics dataframe to long format
performance_metrics_long <- reshape2::melt(performance_metrics, id.vars = "Model")

# Plot the bar chart for performance metrics
plot_perf_metrics <- ggplot(performance_metrics_long, aes(x = variable, y = value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)),
            position = position_dodge(width = 0.9),
            vjust = -0.5,
            color = "black",
            size = 3.5) +
  labs(x = "Metric", y = "Value", fill = "Model") +
  ggtitle("Performance Metrics Comparison") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

### Plot MSE and MAE ###
mse_mae <- data.frame(
  Model = c("Original", "RFE", "Tuned"),
  MSE = c(mse_original, mse_rfe, mse_tuned),
  MAE = c(mae_original, mae_rfe, mae_tuned)
)

# Reshape MSE and MAE dataframe to long format
mse_mae_long <- reshape2::melt(mse_mae, id.vars = "Model")

# Plot the bar chart for MSE and MAE
plot_mse_mae <- ggplot(mse_mae_long, aes(x = variable, y = value, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge") +
  geom_text(aes(label = round(value, 2)),
            position = position_dodge(width = 0.9),
            vjust = -0.5,
            color = "black",
            size = 3.5) +
  labs(x = "Metric", y = "Value", fill = "Model") +
  ggtitle("MSE and MAE Comparison") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5))

# Print the performance metrics comparison
cat("Performance Metrics Comparison:\n")
print(performance_metrics)

# Print the MSE and MAE comparison
cat("MSE and MAE Comparison:\n")
print(mse_mae)

# Display the plots
print(plot_perf_metrics)
print(plot_mse_mae)