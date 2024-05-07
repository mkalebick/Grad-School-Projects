install.packages('MASS')
install.packages('ggplot2')
install.packages('rattle')

# Load libraries 
library(MASS)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(car)
library(readxl)
library(e1071)

# Loading data
bbbc_train <- read_excel("BBBC-Train.xlsx")
bbbc_test <- read_excel("BBBC-Test.xlsx")

# Take a look at the data
names(bbbc_train)
str(bbbc_train)

# Change characters to factors
bbbc_train$Choice = as.factor(bbbc_train$Choice)
bbbc_train$Gender = as.factor(bbbc_train$Gender)

bbbc_test$Choice = as.factor(bbbc_test$Choice)
bbbc_test$Gender = as.factor(bbbc_test$Gender)

# Remove Observation variable
bbbc_train <- dplyr::select(bbbc_train, -Observation)
bbbc_test <- dplyr::select(bbbc_test, -Observation)

# Linear Model
resultsLM = lm(as.numeric(Choice) ~ ., data = bbbc_train)
summary(resultsLM)

predLM = predict(resultsLM, bbbc_test)

# Plot linear regression
ggplot(data = bbbc_test, aes(x = as.numeric(Choice), y = predLM)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red") +  # Adding a reference line
  labs(title = "Linear Regression: Predicted vs Actual Values",
       x = "Actual Values",
       y = "Predicted Values") +
  theme_minimal()

## Resample with more balanced train data 
train_yes = bbbc_train %>% filter(Choice == "1")
train_no = bbbc_train %>% filter(Choice == "0")
set.seed(1)
sample_no_train = sample_n(train_no, nrow(train_yes), replace = TRUE) #matching the same row number
train_bal = rbind(train_yes,sample_no_train) #combining both so they will have a balance data


## Resample with more balanced test data 
test_yes = bbbc_test %>% filter(Choice == "1")
test_no = bbbc_test %>% filter(Choice == "0")
set.seed(1)
sample_no_test = sample_n(test_no, nrow(test_yes), replace = TRUE) #matching the same row number
test_bal = rbind(test_yes,sample_no_test) #combining both so they will have a balance data


###############  SVM  ###############
set.seed(1) ; tuned = tune.svm(Choice ~ ., data = bbbc_train, gamma = seq(.01, .1, by = .01),
                               cost = seq(1, 10, by = .1))

tuned$best.parameters # Gamma: 0.03  # Cost 6.7


# SVM - Kernel: RBF
rbf_SVM = svm(Choice ~ ., data = bbbc_train, gamma = tuned$best.parameters$gamma,
              cost = tuned$best.parameters$cost)
summary(rbf_SVM)

rbf_svm_predict = predict(rbf_SVM, bbbc_test, type = "response")
table(pred = rbf_svm_predict, true = bbbc_test$Choice)
caret::confusionMatrix(as.factor(rbf_svm_predict), as.factor(bbbc_test$Choice), positive = "1")

#### Cost Range: 1-10 # Optimal: 6.7
# Accuracy    : 0.9052
# Sensitivity : 0.9647         
# Specificity : 0.2941

# SVM - Kernel: Linear
linear_SVM = svm(Choice ~ ., data = bbbc_train, gamma = tuned$best.parameters$gamma,
                 cost = tuned$best.parameters$cost, kernel = "linear")

summary(linear_SVM)

linear_svm_predict = predict(linear_SVM, bbbc_test, type = "response")
table(pred = linear_svm_predict, true = bbbc_test$Choice)
caret::confusionMatrix(as.factor(linear_svm_predict), as.factor(bbbc_test$Choice), positive = "1")

# SVM - Kernel: Polynomial
poly_SVM = svm(Choice ~ ., data = bbbc_train, gamma = tuned$best.parameters$gamma,
               cost = tuned$best.parameters$cost, kernel = "polynomial")

summary(poly_SVM)

poly_svm_predict = predict(poly_SVM, bbbc_test, type = "response")
table(pred = poly_svm_predict, true = bbbc_test$Choice)
caret::confusionMatrix(as.factor(poly_svm_predict), as.factor(bbbc_test$Choice), positive = "1")

# SVM - Kernel: Sigmoid 
sigmoid_SVM = svm(Choice ~ ., data = bbbc_train, gamma = tuned$best.parameters$gamma,
                  cost = tuned$best.parameters$cost, kernel = "sigmoid")

summary(sigmoid_SVM)

sigmoid_svm_predict = predict(sigmoid_SVM, bbbc_test, type = "response")
table(pred = sigmoid_svm_predict, true = bbbc_test$Choice)
caret::confusionMatrix(as.factor(sigmoid_svm_predict), as.factor(bbbc_test$Choice), positive = "1")
##Logistic Modeling ##

str(bbbc_train)
m1 = glm(formula = Choice ~ .,data = bbbc_train, family = binomial)
summary(m1)
vif(m1)

#remove Last_purchase
m1 = glm(formula = Choice ~ . -Last_purchase,data = bbbc_train, family = binomial)
summary(m1)
vif(m1)

#model
predprob_log <- predict.glm(m1, bbbc_test, type = "response")
predclass_log = ifelse(predprob_log >= 0.5,yes = 1,0)

# Confusion matrix
caret::confusionMatrix(as.factor(predclass_log), bbbc_test$Choice, positive = "1")
#Accuracy: 0.8957
#Sensitivity: 0.3578
#Specificity: 0.94800


######  Stepwise Selection with AIC
null_model = glm(Choice ~ 1, data = bbbc_test, family = binomial) 
full_model = m1 

step.model.AIC = step(null_model, scope = list(upper = full_model),
                      direction = "both", test = "Chisq", trace = F) 
summary(step.model.AIC) 

# Best model based on stepwise 
m2 <- glm(Choice ~ P_Art+Frequency+Gender+P_DIY+P_Cook+P_Child+First_purchase+Amount_purchased, bbbc_train, family = binomial)
summary(m2)

# Predict the responses on the testing data
bbbc_test$PredProb = predict.glm(m2, newdata = bbbc_test, type = "response") 
bbbc_test$PredY = ifelse(bbbc_test$PredProb >= 0.5, 1,0)

caret::confusionMatrix(as.factor(bbbc_test$Choice), as.factor(bbbc_test$PredY), positive = "1")
# Accuracy    : 0.8957
# Sensitivity : 0.4000         
# Specificity : 0.9377

###################

##### Sending to all:
cost_all = 50000 * 0.65
# 50,000 * Unit mailing cost

revenue_all = 0.0903 * 50000 * (31.95-15-6.75)
# Revenue: 50,000 * 9.03% * (Selling Price - Costs)

profit_all = (revenue_all - cost_all)
profit_all # 13,553


##### Using m1 to target customers:
cost_m1 = (109+73)/2300 * 50000 * 0.65
#         (FP+TP)/2300 * 50,000 * Unit mailing cost
#                ^
#   Proportion of 1’s in model

revenue_m1 = 73/2300 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Costs)

profit_m1 = (revenue_m1 - cost_m1)
profit_m1 # 13,615.22


##### Using m2 to target customers:
cost_m2 = (132+72)/2300 * 50000 * 0.65
#         (FP+TP)/2300 * 50,000 * Unit mailing cost
#                ^
#   Proportion of 1’s in model

revenue_m2 = 72/2300 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Costs)

profit_m2 = (revenue_m2 - cost_m2)
profit_m2 # 13,082.61


##### Using SVM to target customers:
cost_svm = (74+60)/2300 * 50000 * 0.65
#         (FP+TP)/2300 * 50,000 * Unit mailing cost
#                ^
#   Proportion of 1’s in model

revenue_svm = 60/2300 * 50000 * (31.95-15-6.75)
# Number of TP/Total test observations * 50,000 * (Selling Price - Costs)

profit_svm = (revenue_svm - cost_svm)
profit_svm # 13,081.52
