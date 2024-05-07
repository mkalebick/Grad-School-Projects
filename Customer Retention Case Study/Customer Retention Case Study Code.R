############################ Case Study 3 ###############################


# Regression AND classification task
# 1. Predict acquisition (classification)
# 2. See who is required
# 3. Predict duration (regression)


# Loading libraries
library(tidyverse) ; library(SMCRM) ; library(tree) ; library(caret) ;
library(ModelMetrics) ;library(e1071) ; library(corrplot) ; library(ggplot2)
library(MASS) ; library(randomForestSRC) ; library(car)

# Reading in data
data("acquisitionRetention")
df <- acquisitionRetention

###### Data Cleaning & Exploration
str(df)
summary(df)

# Removing customer number since it's just an ID
df = dplyr::select(df, -customer)

# Checking for NAs (there are none)
sum(is.na(df))

# Looking at the relationships between the numeric variables
par(mfrow = c(1, 1))
corrplot(cor(df), method = c("number"), type = c("lower"), number.cex = 0.65)

# Converting variables to factor
df$acquisition <- as.factor(df$acquisition)
df$industry <- as.factor(df$industry)

# Quite a few have high correlation with acquisition:
# duration, profit, ret_exp, freq, crossbuy, sow

# Creating boxplots of the highly correlated variables
par(mfrow = c(3, 3))
boxplot(duration ~ acquisition, data = df, xlab = "Acquisition", ylab = "Duration")
boxplot(profit ~ acquisition, data = df, xlab = "Acquisition", ylab = "Profit")
boxplot(ret_exp ~ acquisition, data = df, xlab = "Acquisition", ylab = "Ret_exp")
boxplot(ret_exp_sq ~ acquisition, data = df, xlab = "Acquisition", ylab = "Ret_exp_sq")
boxplot(freq ~ acquisition, data = df, xlab = "Acquisition", ylab = "Freq")
boxplot(freq_sq ~ acquisition, data = df, xlab = "Acquisition", ylab = "Freq_sq")
boxplot(crossbuy ~ acquisition, data = df, xlab = "Acquisition", ylab = "Crossbuy")
boxplot(sow ~ acquisition, data = df, xlab = "Acquisition", ylab = "Sow")
# When acquisition is zero, the variables above are also zero (except profit,
# which is negative by definition)

# Removing perfectly separated variables as identified above
df1 <- df %>% subset(select = c(-duration, -profit, -ret_exp, -ret_exp_sq, -freq,
                               -freq_sq, -crossbuy, -sow))

pairs(df1) #relationships


##### Splitting into train and test
set.seed(123)
index = sample(nrow(df1), 0.8*nrow(df1), replace = F) # 80/20 split
train = df1[index,]
test = df1[-index,]

# Checking balance of data
summary(train$acquisition)
summary(test$acquisition)


###### Modeling for Acquisition (Classification Task)

### Logistic regression
glm.all = glm(acquisition ~ ., data = train, family = binomial)
summary(glm.all)
vif(glm.all)

# Removing acq_exp due to multicollinearity
glm.fit = glm(acquisition ~ .-acq_exp, data = train, family = binomial)
summary(glm.fit)
vif(glm.fit)

# Computing probabilities and getting predictions
glm.probs <- predict.glm(glm.fit, test, type = "response")
glm.preds = ifelse(glm.probs >= 0.5, yes = 1, 0)

# Confusion matrix
caret::confusionMatrix(as.factor(glm.preds), test$acquisition, positive = "1")
# Accuracy    : 0.81
# Sensitivity : 0.9688          
# Specificity : 0.5278 


### Decision Tree
tree.fit <- tree(acquisition ~ ., data = train)
summary(tree.fit)

# Plotting the tree 
par(mfrow = c(1, 1))
plot(tree.fit)
text(tree.fit, cex = 0.6)

# Getting predictions
tree.preds = predict(tree.fit, test, type = "class")
caret::confusionMatrix(as.factor(tree.preds), test$acquisition, positive = "1")
# Accuracy    : 0.78
# Sensitivity : 0.8750          
# Specificity : 0.6111



### Random Forest (with manual tuning of parameters)

set.seed(123) ; rf_model1 = rfsrc(acquisition ~ ., data = train,
                                  importance = TRUE, ntree = 1000,
                                  nodesize = 9, nodedepth = 5)

rf_model1
rf1.preds = predict(rf_model1, test)$class
caret::confusionMatrix(as.factor(rf1.preds), test$acquisition, positive = "1")
# Accuracy    : 0.81
# Sensitivity : 0.9219          
# Specificity : 0.6111



###### Random Forest Plots: Variable Importance, Minimal Depth & Interactions

### Variable Importance
importance_df <- as.data.frame(rf_model1$importance) %>%
  tibble::rownames_to_column(var = "variable")

importance_df %>% 
  ggplot(aes(x = reorder(variable, all), y = all)) +
  geom_bar(stat = "identity", fill = "cornflowerblue", color = "black") +
  coord_flip() +
  labs(x = "Variables", y = "Variable importance") +
  theme_minimal()


### Minimal Depth
# Helps identify the most important variables in the random forest model by
# indicating how close they are to the root of the trees. Lower minimal depth
# values suggest greater importance or relevance of the variables in predicting
# the outcome.

mindepth <- max.subtree(rf_model1, sub.order = TRUE)

# Print first order depths
print(round(mindepth$order, 3)[, 1])

# Plot minimal depth
md_df <- data.frame(md = round(mindepth$order, 3)[, 1]) %>%
  tibble::rownames_to_column(var = "variable")

ggplot(md_df, aes(x = reorder(variable, desc(md)), y = md)) +
  geom_bar(stat = "identity", fill = "cornflowerblue", color = "black", width = 0.2) +
  coord_flip() +
  labs(x = "Variables", y = "Minimal Depth") +
  theme_minimal()

### Interactions
# Cross-checking with variable importance
find.interaction(rf_model1, method = "vimp", importance = "permute")
# We found that the most significant interaction terms are:
   # employees:industry
   # employees:revenue
   # employees:acq_exp_sq
   # employees:acq_exp
# We will test how including these affects the model's accuracy below.


### Tuned Random Forest for `Acquisition` (Using grid search & interaction terms)

# Establishing list of possible values for hyper-parameters
mtry.values <- seq(2,5,1)
nodesize.values <- seq(4,10,1)
#nodedepth.values <- seq(1,9,1)
ntree.values <- seq(1000,9000,2000)

# Creating data frame containing all combinations 
hyper_grid = expand.grid(mtry = mtry.values,
                         nodesize = nodesize.values,
                         #nodedepth = nodedepth.values,
                         ntree = ntree.values)

# Creating empty vector to store out-of-bag error values
oob_err = c()

# Looping over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  # Training Random Forest
  set.seed(123) ; model = rfsrc(acquisition ~ .+employees*industry, 
                                 data = train, importance = TRUE,
                                 mtry = hyper_grid$mtry[i],
                                 nodesize = hyper_grid$nodesize[i],
                                 #nodedepth = hyper_grid$nodedepth[i],
                                 ntree = hyper_grid$ntree[i])  
  # Storing OOB error for the model                      
  oob_err[i] <- model$err.rate[length(model$err.rate)]
}

# Identifying optimal set of hyper-parameters based on OOB error
opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])

set.seed(123) ; rf1.tuned = rfsrc(acquisition ~ .+employees*industry,
                                 data = train, importance = TRUE,
                                 mtry = 3, nodesize = 8, ntree = 7000)
rf1.tuned
# Note: The settings in the model above yielded the best accuracy (0.80) among
# all the combinations of interaction terms and hyper-parameters we tested.

# Getting predictions
rf1.tuned.preds = predict(rf1.tuned, test)$class
caret::confusionMatrix(as.factor(rf1.tuned.preds), test$acquisition, positive = "1")
# Accuracy    : 0.8
# Sensitivity : 0.9062          
# Specificity : 0.6111
# After testing with different numbers of interaction terms (0, 1, 2, 3, 4),
# as well as different parameters in our hyper grid, we did not achieve an
# improvement in model accuracy. The grid search is outputting the best parameter
# values based on minimal out-of-bag error, and while this metric provides
# an estimate of how well the random forest model is likely to perform on unseen
# data, it is not guaranteed that the same parameter settings will perform best
# on our specific testing set. Nonetheless, using grid search to tune parameters
# should generally improve results, and its worth looking further into this for
# future scenarios, for instance by testing with different seeds and/or other
# ratios for the train/test split.



##### Preparing Data for Regression Task

# Creating temporary df combining predictions and original df
temp_df = cbind(df, rf1.preds)

# Filtering predictions for a value of 1 (i.e. successful customer acquisition)
df2 = temp_df %>%
  filter(rf1.preds == 1 & acquisition = 1) %>%
  subset(select = c(-acquisition, -rf1.preds))

# Correlation plot
par(mfrow = c(1, 1))
df2 %>%
  select_if(is.numeric) %>%
  cor() %>%
  corrplot(method = "number", type = "lower", number.cex = 0.65)

pairs(df2)

# Removing highly correlated variables
df2 = df2 %>%
  subset(select = c(-profit, -ret_exp))

# Splitting into train and test sets
set.seed(123)
index = sample(nrow(df2), 0.8*nrow(df2), replace = F) # 80/20 split
train_new = df2[index,]
test_new = df2[-index,]



###### Modeling for Duration (Regression Task)

### Random Forest
set.seed(123) ; rf_model2 <- rfsrc(duration ~ ., train_new, ntree = 1000)
rfsrc.summary(rf_model2)

# Predicting on the test set and computing MSE
rf2.preds = predict(rf_model2, test_new)
mean((test_new$duration - rf2.preds$predicted)^2)
# MSE: 2385.798


##### Tuned Random Forest for `Duration` (Using grid search)

# Establishing list of possible values for hyper-parameters
mtry.values <- seq(2,10,1)
nodesize.values <- seq(4,10,2)
#nodedepth.values <-seq(2,8,2)
ntree.values <- seq(1000,9000,2000)

# Creating data frame containing all combinations 
hyper_grid = expand.grid(mtry = mtry.values,
                         nodesize = nodesize.values,
                         #nodedepth = nodedepth.values,
                         ntree = ntree.values)

# Creating empty vector to store out-of-bag error values
oob_err = c()

# Looping over the rows of hyper_grid to train the grid of models
for (i in 1:nrow(hyper_grid)) {
  # Training Random Forest
  set.seed(123) ; model = rfsrc(duration ~ ., 
                                 data = train_new,
                                 mtry = hyper_grid$mtry[i],
                                 nodesize = hyper_grid$nodesize[i],
                                 #nodedepth = hyper_grid$nodedepth[i],
                                 ntree = hyper_grid$ntree[i])  
  # Storing OOB error for the model                      
  oob_err[i] <- model$err.rate[length(model$err.rate)]
}

# Identifying optimal set of hyper-parameters based on OOB error
opt_i <- which.min(oob_err)
print(hyper_grid[opt_i,])

# Applying tuned params to the model
set.seed(123) ; rf2.tuned = rfsrc(duration ~ ., data = train_new,
                                 importance = TRUE, ntree = 5000,
                                 mtry = 8, nodesize = 4)
rf2.tuned

rf2.tuned.preds = predict(rf2.tuned, test_new)
mean((test_new$duration - rf2.tuned.preds$predicted)^2)
# MSE: 856.1437
# For this task, we find that using grid search significantly improves the test
# accuracy, decreasing the MSE from 2385.8 to 856.1.




###### Extra credit: Generate PDP plots for all variables

plot.variable(rf_model1, partial = TRUE)

# Partial Dependence Plots (PDP) can show: 

# Feature Importance: It helps visualize the importance of a particular feature
# in the model's predictions. A steeper slope indicates a stronger influence of
# that feature on the predicted outcome.

# Effect of the Feature: It shows how the predicted outcome changes as the value
# of the feature varies, while holding other features constant at their average
# values or at specified values.

# Directionality: It indicates whether the feature has a positive or negative
# effect on the predicted outcome. For example, if the PD plot slopes upward,
# it suggests that increasing the feature value tends to increase the predicted
# outcome, and vice versa.

# Non-linear Relationships: It can reveal non-linear relationships between the
# feature and the predicted outcome. The shape of the PD plot can provide
# insights into how the feature interacts with other variables in the model.




