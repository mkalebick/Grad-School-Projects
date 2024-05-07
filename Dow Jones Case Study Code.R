############################ Case Study 3 ###############################

# Loading libraries
library(tidyverse) ; library(dplyr) ; library(lubridate) ; library(car) ;
library(caret) ; library(ModelMetrics) ; library(tree) ; library(e1071) ;
library(readxl) ; library(quantmod) ; library(tseries) ; library(ggplot2)

# Reading in data
setwd("~/Desktop/Applications/Data")
dow_jones <- read.csv("dow_jones_index.data")

###### Data Cleaning 

# Cleaning columns with "$"
dow_jones$open = as.numeric(gsub("\\$", "", dow_jones$open))
dow_jones$high = as.numeric(gsub("\\$", "", dow_jones$high))
dow_jones$low = as.numeric(gsub("\\$", "", dow_jones$low))
dow_jones$close = as.numeric(gsub("\\$", "", dow_jones$close))

# Removing NAs
dow_jones <- na.omit(dow_jones)

# Creating `week` column
data <- dow_jones %>%
  arrange(stock) %>% 
  group_by(stock) %>%
  mutate(week = seq(1, n(), by = 1)) %>%
  ungroup()


# Removing "next week" columns and `date` column
data <- data %>%
  subset(select = c(-next_weeks_open, -next_weeks_close, -date))


##### Exploring lags

lag.plot(data$percent_change_next_weeks_price, pch = ".", set.lags = 1:4)
# We are not seeing any impact of using lag on the response variable



##### Train/Test Split

# Splitting into train (quarter 1) and test (quarter 2)
train <- data %>% 
  filter(quarter == 1) %>% 
  select(-quarter)
test <- data %>% 
  filter(quarter == 2) %>% 
  select(-quarter)


##### Modeling

### Linear Model:

# Checking for multicollinearity
vif_check <- lm(percent_change_next_weeks_price ~ low + volume +
                 percent_change_price + percent_change_volume_over_last_wk + 
                 days_to_next_dividend + percent_return_next_dividend +
                 previous_weeks_volume, data = train)
vif(vif_check) # Removed variables: close, high, open


# Getting unique stock values
unique_stocks <- as.factor(unique(train$stock))

# Creating data frame for performance metric (RMSE)
RMSE = rep(NA, length(unique_stocks)) 
lm.results = data.frame("Stock" = unique_stocks, "RMSE" = RMSE)

# Looping through each stock in the train set
for(i in 1:length(unique_stocks)) {
  # Subsetting train and test data set for current stock
  subset_train <- train %>% 
    subset(stock == unique_stocks[i])
  subset_test <- test %>% 
    subset(stock == unique_stocks[i])
  
  # Fitting the linear regression model using subset data
  lm.fit <- lm(percent_change_next_weeks_price ~ low + volume +
                   percent_change_price + percent_change_volume_over_last_wk + 
                   days_to_next_dividend + percent_return_next_dividend +
                   previous_weeks_volume, data = subset_train)
  
  lm.preds = predict(lm.fit, subset_test) 
  
  # Calculating RMSE
  lm.rmse <- rmse(subset_test$percent_change_next_weeks_price, lm.preds)
  lm.results[i, "RMSE"] <- lm.rmse
}

#print(lm.results)



### SVR:

# Initialize a list to store predictions for each stock
predictions <- list()

svr.results = data.frame("Stock" = unique_stocks, "RMSE" = RMSE)

for(i in 1:length(unique_stocks)) {
  subset_train <- train %>% 
    subset(stock == unique_stocks[i])
  subset_test <- test %>% 
    subset(stock == unique_stocks[i])
  
  set.seed(123) ; tuned = tune.svm(percent_change_next_weeks_price ~ low + volume +
                                     percent_change_price + percent_change_volume_over_last_wk +
                                     days_to_next_dividend + percent_return_next_dividend +
                                     previous_weeks_volume, data = subset_train,
                                   gamma = seq(.01, .1, by = .01),
                                   cost = seq(.1, 1, by = .1), scale = TRUE)
  
  svr.fit <- svm(percent_change_next_weeks_price ~ low + volume +
                   percent_change_price + percent_change_volume_over_last_wk +
                   days_to_next_dividend + percent_return_next_dividend +
                   previous_weeks_volume, data = subset_train,
                 gamma = tuned$best.parameters$gamma, cost = tuned$best.parameters$cost)
  
  svr.preds = predict(svr.fit, subset_test, type = "response") 
  
  svr.rmse <- sqrt(mean((svr.preds - subset_test$percent_change_next_weeks_price)^2))
  svr.results[i, "RMSE"] <- svr.rmse
  
  # Storing predictions for the current stock
  predictions[[i]] <- svr.preds
  
}

#print(svr.results)
#print(last_prediction)

### Decision Tree:

tree.results = data.frame("Stock" = unique_stocks, "RMSE" = RMSE)

for(i in 1:length(unique_stocks)) {
  subset_train <- train %>% 
    subset(stock == unique_stocks[i])
  subset_test <- test %>% 
    subset(stock == unique_stocks[i])
  
  tree.fit <- tree(percent_change_next_weeks_price ~ low + volume +
                     percent_change_price + percent_change_volume_over_last_wk + 
                     days_to_next_dividend + percent_return_next_dividend +
                     previous_weeks_volume, data = subset_train)
  
  tree.preds = predict(tree.fit, subset_test) 
  
  tree.rmse <- rmse(subset_test$percent_change_next_weeks_price, tree.preds)
  tree.results[i, "RMSE"] <- tree.rmse
  
}

#print(tree.results)


##### Comparing models
mean(lm.results$RMSE)
mean(svr.results$RMSE)
mean(tree.results$RMSE)


# CAPM: Get S&P500 data from yahoo.finance
SP <- read_excel("SP500.xlsx")
SP <- SP %>% arrange(Date)
week = seq(from = 1, to = dim(SP)[1], by = 1)
SP <- cbind(week, SP)

# Computing the returns on the close price for SP500 and each stock in Dow Jones
ReturnSP = Delt(SP[,6])

ReturnAA <- data %>% filter(stock == "AA") %>% pull(close) %>% Delt()
ReturnAXP <- data %>% filter(stock == "AXP") %>% pull(close) %>% Delt()
ReturnBA <- data %>% filter(stock == "BA") %>% pull(close) %>% Delt()
ReturnBAC <- data %>% filter(stock == "BAC") %>% pull(close) %>% Delt()
ReturnCAT <- data %>% filter(stock == "CAT") %>% pull(close) %>% Delt()
ReturnCSCO <- data %>% filter(stock == "CSCO") %>% pull(close) %>% Delt()
ReturnCVX <- data %>% filter(stock == "CVX") %>% pull(close) %>% Delt()
ReturnDD <- data %>% filter(stock == "DD") %>% pull(close) %>% Delt()
ReturnDIS <- data %>% filter(stock == "DIS") %>% pull(close) %>% Delt()
ReturnGE <- data %>% filter(stock == "GE") %>% pull(close) %>% Delt()
ReturnHD <- data %>% filter(stock == "HD") %>% pull(close) %>% Delt()
ReturnHPQ <- data %>% filter(stock == "HPQ") %>% pull(close) %>% Delt()
ReturnIBM <- data %>% filter(stock == "IBM") %>% pull(close) %>% Delt()
ReturnINTC <- data %>% filter(stock == "INTC") %>% pull(close) %>% Delt()
ReturnJNJ <- data %>% filter(stock == "JNJ") %>% pull(close) %>% Delt()
ReturnJPM <- data %>% filter(stock == "JPM") %>% pull(close) %>% Delt()
ReturnKO <- data %>% filter(stock == "KO") %>% pull(close) %>% Delt()
ReturnKRFT <- data %>% filter(stock == "KRFT") %>% pull(close) %>% Delt()
ReturnMCD <- data %>% filter(stock == "MCD") %>% pull(close) %>% Delt()
ReturnMMM <- data %>% filter(stock == "MMM") %>% pull(close) %>% Delt()
ReturnMRK <- data %>% filter(stock == "MRK") %>% pull(close) %>% Delt()
ReturnMSFT <- data %>% filter(stock == "MSFT") %>% pull(close) %>% Delt()
ReturnPFE <- data %>% filter(stock == "PFE") %>% pull(close) %>% Delt()
ReturnPG <- data %>% filter(stock == "PG") %>% pull(close) %>% Delt()
ReturnT <- data %>% filter(stock == "T") %>% pull(close) %>% Delt()
ReturnTRV <- data %>% filter(stock == "TRV") %>% pull(close) %>% Delt()
ReturnUTX <- data %>% filter(stock == "UTX") %>% pull(close) %>% Delt()
ReturnWMT <- data %>% filter(stock == "WMT") %>% pull(close) %>% Delt()
ReturnVZ <- data %>% filter(stock == "VZ") %>% pull(close) %>% Delt()
ReturnXOM <- data %>% filter(stock == "XOM") %>% pull(close) %>% Delt()

# Combining data
All_Returns = cbind(ReturnSP,
               ReturnAA,
               ReturnAXP,
               ReturnBA, 
               ReturnBAC,
               ReturnCAT,
               ReturnCSCO,
               ReturnCVX,
               ReturnDD,
               ReturnDIS,
               ReturnGE,
               ReturnHD,
               ReturnHPQ,
               ReturnIBM,
               ReturnINTC,
               ReturnJNJ,
               ReturnJPM,
               ReturnKO,
               ReturnKRFT,
               ReturnMCD,
               ReturnMMM,
               ReturnMRK,
               ReturnMSFT,
               ReturnPFE,
               ReturnPG,
               ReturnT,
               ReturnTRV,
               ReturnUTX,
               ReturnWMT,
               ReturnVZ,
               ReturnXOM)

colnames(All_Returns) = c("SP500", "AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "KRFT", "MCD", "MMM", "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UTX", "WMT", "VZ", "XOM")
#head(All_Returns)
All_Returns <- na.omit(All_Returns)


# Creating boxplot to show volatility of all individual stocks and S&P 500.
all_stocks <- c(unique_stocks, "SP500")

All_Returns_long <- tidyr::gather(data.frame(All_Returns), key = "Stock", value = "Return")

sp500_fill <- "salmon"

boxplot <- ggplot(All_Returns_long, aes(x = Stock, y = Return,
                                        fill = ifelse(Stock == "SP500", "SP500", "Others"))) +
  geom_boxplot() +
  scale_fill_manual(values = c("Others" = "#CCCCFF", "SP500" = sp500_fill)) +
  labs(x = "Stock", y = "Return", title = "Expected Return") +
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  theme(plot.title = element_text(hjust = 0.5)) +
  guides(fill = FALSE)

print(boxplot)
# We see that the risk is much lesser in the S&P compared to the individual stocks. 

# Computing mean and standard deviation for the returns.
DataMean = apply(All_Returns, 2, mean)
DataSD = apply(All_Returns, 2, sd)
cbind(DataMean, DataSD)

# Fitting linear models to get betas
lm.AA <-  lm(AA ~ SP500, data = as.data.frame(All_Returns))
BetaAA <- summary(lm.AA)$coefficients[2, 1]

lm.AXP <- lm(AXP ~ SP500, data = as.data.frame(All_Returns))
BetaAXP <- summary(lm.AXP)$coefficients[2, 1]

lm.BA <- lm(BA ~ SP500, data = as.data.frame(All_Returns))
BetaBA <- summary(lm.BA)$coefficients[2, 1]

lm.BAC <- lm(BAC ~ SP500, data = as.data.frame(All_Returns))
BetaBAC <- summary(lm.BAC)$coefficients[2, 1]

lm.CAT <- lm(CAT ~ SP500, data = as.data.frame(All_Returns))
BetaCAT <- summary(lm.CAT)$coefficients[2, 1]

lm.CSCO <- lm(CSCO ~ SP500, data = as.data.frame(All_Returns))
BetaCSCO <- summary(lm.CSCO)$coefficients[2, 1]

lm.CVX <- lm(CVX ~ SP500, data = as.data.frame(All_Returns))
BetaCVX <- summary(lm.CVX)$coefficients[2, 1]

lm.DD <- lm(DD ~ SP500, data = as.data.frame(All_Returns))
BetaDD <- summary(lm.DD)$coefficients[2, 1]

lm.DIS <- lm(DIS ~ SP500, data = as.data.frame(All_Returns))
BetaDIS <- summary(lm.DIS)$coefficients[2, 1]

lm.GE <- lm(GE ~ SP500, data = as.data.frame(All_Returns))
BetaGE <- summary(lm.GE)$coefficients[2, 1]

lm.HD <- lm(HD ~ SP500, data = as.data.frame(All_Returns))
BetaHD <- summary(lm.HD)$coefficients[2, 1]

lm.HPQ <- lm(HPQ ~ SP500, data = as.data.frame(All_Returns))
BetaHPQ <- summary(lm.HPQ)$coefficients[2, 1]

lm.IBM <- lm(IBM ~ SP500, data = as.data.frame(All_Returns))
BetaIBM <- summary(lm.IBM)$coefficients[2, 1]

lm.INTC <- lm(INTC ~ SP500, data = as.data.frame(All_Returns))
BetaINTC <- summary(lm.INTC)$coefficients[2, 1]

lm.JNJ <- lm(JNJ ~ SP500, data = as.data.frame(All_Returns))
BetaJNJ <- summary(lm.JNJ)$coefficients[2, 1]

lm.JPM <- lm(JPM ~ SP500, data = as.data.frame(All_Returns))
BetaJPM <- summary(lm.JPM)$coefficients[2, 1]

lm.KO <- lm(KO ~ SP500, data = as.data.frame(All_Returns))
BetaKO <- summary(lm.KO)$coefficients[2, 1]

lm.KRFT <- lm(KRFT ~ SP500, data = as.data.frame(All_Returns))
BetaKRFT <- summary(lm.KRFT)$coefficients[2, 1]

lm.MCD <- lm(MCD ~ SP500, data = as.data.frame(All_Returns))
BetaMCD <- summary(lm.MCD)$coefficients[2, 1]

lm.MMM <- lm(MMM ~ SP500, data = as.data.frame(All_Returns))
BetaMMM <- summary(lm.MMM)$coefficients[2, 1]

lm.MRK <- lm(MRK ~ SP500, data = as.data.frame(All_Returns))
BetaMRK <- summary(lm.MRK)$coefficients[2, 1]

lm.MSFT <- lm(MSFT ~ SP500, data = as.data.frame(All_Returns))
BetaMSFT <- summary(lm.MSFT)$coefficients[2, 1]

lm.PFE <- lm(PFE ~ SP500, data = as.data.frame(All_Returns))
BetaPFE <- summary(lm.PFE)$coefficients[2, 1]

lm.PG <- lm(PG ~ SP500, data = as.data.frame(All_Returns))
BetaPG <- summary(lm.PG)$coefficients[2, 1]

lm.T <- lm(T ~ SP500, data = as.data.frame(All_Returns))
BetaT <- summary(lm.T)$coefficients[2, 1]

lm.TRV <- lm(TRV ~ SP500, data = as.data.frame(All_Returns))
BetaTRV <- summary(lm.TRV)$coefficients[2, 1]

lm.UTX <- lm(UTX ~ SP500, data = as.data.frame(All_Returns))
BetaUTX <- summary(lm.UTX)$coefficients[2, 1]

lm.WMT <- lm(WMT ~ SP500, data = as.data.frame(All_Returns))
BetaWMT <- summary(lm.WMT)$coefficients[2, 1]

lm.VZ <- lm(VZ ~ SP500, data = as.data.frame(All_Returns))
BetaVZ <- summary(lm.VZ)$coefficients[2, 1]

lm.XOM <- lm(XOM ~ SP500, data = as.data.frame(All_Returns))
BetaXOM <- summary(lm.XOM)$coefficients[2, 1]

# Create a vector of stock names
stock_names <- c("AA", "AXP", "BA", "BAC", "CAT", "CSCO", "CVX", "DD", "DIS", "GE", "HD",
                 "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "KRFT", "MCD", "MMM", 
                 "MRK", "MSFT", "PFE", "PG", "T", "TRV", "UTX", "WMT", "VZ", "XOM")

# Create a vector of corresponding beta coefficients
beta_values <- c(BetaAA, BetaAXP, BetaBA, BetaBAC, BetaCAT, BetaCSCO, BetaCVX, BetaDD, BetaDIS, BetaGE, BetaHD,
                 BetaHPQ, BetaIBM, BetaINTC, BetaJNJ, BetaJPM, BetaKO, BetaKRFT, BetaMCD, BetaMMM, 
                 BetaMRK, BetaMSFT, BetaPFE, BetaPG, BetaT, BetaTRV, BetaUTX, BetaWMT, BetaVZ, BetaXOM)

# Create a data frame
beta_df <- data.frame(Stock = stock_names, Beta = beta_values)

# Display the data frame
print(beta_df)


###### Comparing Risk and Return for Each Stock

# Extracting predictions for week 13 for each stock and creating a df for week 14
week_13_predictions <- sapply(predictions, `[`, 13)

week_14_data <- data.frame(
  Stock = unique_stocks,
  Prediction = week_13_predictions,
  stringsAsFactors = FALSE)

# print(week_14_data)

# Merging beta values with week 14 predictions
merged_data <- merge(beta_df, week_14_data, by = "Stock")

# Plotting risk against return
scatterplot <- ggplot(merged_data, aes(x = Beta, y = Prediction)) +
  geom_point(color = "#CCCCFF", size = 3) +
  geom_text(aes(label = Stock), hjust = 0, vjust = 0, size = 3) + 
  labs(x = "Beta (Risk)", y = "Predicted Return", title = "Risk vs. Return") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") + 
  geom_vline(xintercept = 1, linetype = "dashed", color = "red") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5)) 

print(scatterplot)








