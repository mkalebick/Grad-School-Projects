install.packages('MASS')
install.packages('ggplot2')
install.packages('rattle')

# Load libraries 
library(MASS)
library(ggplot2)
library(tidyverse)
library(corrplot)
library(car)

# Load the bank dataset
bank = read.csv('bank-additional.csv', na.strings = "unknown", sep = ';')
bank = na.omit(bank)
bank = dplyr::select(bank, -default) # Remove default because only 1 yes
bank = dplyr::select(bank, -duration) # Remove duration

# Change characters to factors
bank$job = as.factor(bank$job)
bank$marital = as.factor(bank$marital)
bank$default = as.factor(bank$default)
bank$housing = as.factor(bank$housing)
bank$loan = as.factor(bank$loan)
bank$contact = as.factor(bank$contact)
bank$poutcome = as.factor(bank$poutcome)
bank$y = as.factor(bank$y)
bank$contacted = as.factor(bank$contacted)
str(bank)

# Change pdays to levels then add column for if they were contacted
bank$pdays <- cut(bank$pdays, c(0, 7, 14, 998),
                  c("Less than 1 week", "Between 1-2 weeks",
                    "More than 2 weeks"), right = TRUE)

bank$contacted <- ifelse(is.na(bank$pdays), "No", "Yes")

# Plot correlations
bank_num <- dplyr::select_if(bank, is.numeric)
M = cor(bank_num)
corrplot(M, method = c("shade"))

pairs(bank_num)


# dummy variables (marital, housing, loan, contact, poutcome, contacted)
bank$marital_dummy1 = ifelse(bank$marital == 'divorced', 1, 0)
bank$marital_dummy2 = ifelse(bank$marital == 'married', 1, 0)

bank$housing_dummy = ifelse(bank$housing == 'yes', 1, 0)

bank$loan_dummy = ifelse(bank$loan == 'yes', 1, 0)

bank$contact_dummy = ifelse(bank$contact == 'cellular', 1, 0)

bank$poutcome_dummy1 = ifelse(bank$poutcome == 'failure', 1, 0)
bank$poutcome_dummy2 = ifelse(bank$poutcome == 'success', 1, 0)

bank$housing_contacted = ifelse(bank$contacted == 'Yes', 1, 0)

# Split into test and train
set.seed(1)
index <- sample(nrow(bank),0.8*nrow(bank), replace = F) # Setting training sample to be 70% of the data
banktrain <- bank[index,]
banktest <- bank[-index,]

#### m1
m1 = glm(formula = y ~ marital_dummy1 + marital_dummy2 + housing_dummy + loan_dummy + contact_dummy
         + poutcome_dummy1 +poutcome_dummy2 + housing_contacted + age + campaign
         + previous + emp.var.rate + cons.price.idx + cons.conf.idx + euribor3m
         + nr.employed, data = banktrain, family = binomial)
summary(m1)

# Remove euribor3m
m1 = glm(formula = y ~ marital_dummy1 + marital_dummy2 + housing_dummy + loan_dummy + contact_dummy
         + poutcome_dummy1 +poutcome_dummy2 + housing_contacted + age + campaign
         + previous + emp.var.rate + cons.price.idx + cons.conf.idx + nr.employed,
         data = banktrain, family = binomial)

# Remove emp.var.rate
m1 = glm(formula = y ~ marital_dummy1 + marital_dummy2 + housing_dummy + loan_dummy + contact_dummy
         + poutcome_dummy1 + poutcome_dummy2 + housing_contacted + age + campaign
         + previous + cons.price.idx + cons.conf.idx + nr.employed, data = banktrain, family = binomial)

# Remove contacted_dummy
m1 = glm(formula = y ~ marital_dummy1 + marital_dummy2 + housing_dummy + loan_dummy + contact_dummy
         + poutcome_dummy1 + poutcome_dummy2 + age + campaign
         + previous + cons.price.idx + cons.conf.idx + nr.employed, data = banktrain, family = binomial)

vif(m1) # check multicollinearity
cor(bank[sapply(bank,is.numeric)])



#UNBALANCED

# Predict the responses on the unbalanced testing data. 
predprob_log <- predict.glm(m1, banktest, type = "response")
predclass_log = ifelse(predprob_log >= 0.5, "yes", "no")

# Stepwise Selection with AIC
null_model = glm(y ~ 1, data = banktrain, family = binomial) 
full_model = m1

step.model.AIC = step(null_model, scope = list(upper = full_model),
                      direction = "both", test = "Chisq", trace = F) 
summary(step.model.AIC)

#### m2
m2 = glm(formula = y ~ nr.employed + contact_dummy + poutcome_dummy1 + cons.conf.idx
         + campaign + cons.price.idx + poutcome_dummy2, data = banktrain, family = binomial)
summary(m2)

banktest$PredProb = predict.glm(m2, newdata = banktest, type = "response")
banktest$PredY = ifelse(banktest$PredProb >= 0.5, "yes", "no")

caret::confusionMatrix(as.factor(banktest$y), as.factor(banktest$PredY))


#BALANCED

## Resample with more balanced data 
bank_yes_cust = bank %>% filter(y == "yes")
bank_no_cust = bank %>% filter(y == "no")
set.seed(1)
sample_yes_cust = sample_n(bank_yes_cust, nrow(bank_no_cust), replace = TRUE) #matching the same row number
bank_bal = rbind(bank_no_cust,sample_yes_cust) #combining both so they will have a balance data

# Split data into training and testing balanced samples
set.seed(1) 
tr_ind_bal <- sample(nrow(bank_bal),0.8*nrow(bank_bal),replace = F) # Set training sample to be 80% of the data
banktrain_bal <- bank_bal[tr_ind_bal,]
banktest_bal <- bank_bal[-tr_ind_bal,]

### Build model with balanced data
m1_bal = glm(y ~ age + marital_dummy1 + marital_dummy2 + housing_dummy + loan_dummy + 
               contact_dummy + campaign + previous + poutcome_dummy1 + poutcome_dummy2
             + cons.price.idx + cons.conf.idx + nr.employed, data = banktrain_bal, family = binomial)
summary(m1_bal) ## look at results
vif(m1_bal) # double check multicollinearity

# Added removed variables back in to compare
m1_bal = glm(y ~ age + marital_dummy1 + marital_dummy2 + housing_dummy + loan_dummy + 
               contact_dummy + campaign + previous + poutcome_dummy1 + poutcome_dummy2
             + cons.price.idx + cons.conf.idx + nr.employed + euribor3m + emp.var.rate + contacted_dummy, data = banktrain_bal, family = binomial)
summary(m1_bal) ## look at results
vif(m1_bal) # check multicollinearity

# Predict the responses on the balanced testing data. 
predprob_log_bal <- predict.glm(m1_bal, banktest_bal, type = "response")
predclass_log_bal = ifelse(predprob_log_bal >= 0.5, "yes", "no")

# Confusion matrix to compare balanced and unbalanced
caret::confusionMatrix(as.factor(banktest$y), as.factor(banktest$PredY))
caret::confusionMatrix(as.factor(predclass_log_bal), banktest_bal$y, positive = "yes")

## variable selection - reduce complexity if the model
m2_bal = step(glm(y ~ nr.employed + contact_dummy + poutcome_dummy1 + cons.conf.idx
                  + campaign + cons.price.idx + poutcome_dummy2, data = banktrain_bal, family = binomial), direction = "both")
summary(m2_bal) ## look at results
vif(m2_bal) # double check multicollinearity

# Predict the responses on the testing data. 
predprob2_log_bal <- predict.glm(m2_bal, banktest_bal, type = "response")
predclass2_log_bal = ifelse(predprob2_log_bal >= 0.5, "yes", "no")
caret::confusionMatrix(as.factor(predclass2_log_bal), banktest_bal$y, positive = "yes")
