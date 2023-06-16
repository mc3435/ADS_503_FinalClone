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

install.packages("factoextra")

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

# Pearson Correlation
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
var_only <- df_clean[, !(names(df_clean) %in% c("diagnosis"))]

var_pca <- prcomp(var_only, center=TRUE, scale.=TRUE)
plot(var_pca, type="l", main='',color='red')

grid(nx = 10, ny = 14)
title(main = "Principal components weight", sub = NULL, xlab = "Components")
box()

ggbiplot(bc.pca, choices=1:2, obs.scale = 1, var.scale = 1, groups = bc.diag, 
         ellipse = TRUE, circle = TRUE, varname.size = 3, ellipse.prob = 0.68, circle.prob = 0.69) +
  scale_color_discrete(name = 'Diagnosis (B: beningn, M: malignant)') + theme_bw() + 
  labs(title = "Principal Component Analysis", 
       subtitle = "1. Data distribution in the plan of PC1 and PC2\n2. Directions of components in the same plane") +
  theme(legend.direction = 'horizontal', legend.position = 'bottom')