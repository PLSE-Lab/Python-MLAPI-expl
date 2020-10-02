############################ Retail-Giant Sales Forecasting #################################
# 1. Business Understanding
# 2. Data Understanding and EDA
# 3. Data Preparation
# 4. Model Building and evaluation
#  4.1 Classical decomposition
#  4.2 Auto-ARIMA

#####################################################################################

# 1. Business Understanding: 

#The objective is to forecast sales and demand for the next 6 months, 
#which would help manage the revenue and inventory accordingly. The forecast should be
#made for the two most important market buckets (7 market segments * 3 categories)

#####################################################################################

# 2. Data Understanding: 
# Number of rows in train: 51290
# Number of Attributes: 24 

# 3. Data Preparation: 
# Group the data by Market Segment Year and Month
# Aggregate the Sales, Quantity, and Profits over the respective Year and Month
# Find the two most profitable (based on average monthly profits) and consistently
#profitable (coefficient of variation) buckets

# 4. Model building and evaluation:
# Classical decomposition and Auto-ARIMA models will be considered.

#############################################################################################
#Loading Neccessary libraries

library(dplyr)
library(ggplot2)
library(cowplot)
library(forecast)
library(tseries)
require(graphics)

#############################################################################################
#Reading in the global-store data-set, and glancing through it
df <- read.csv("Global Superstore.csv",stringsAsFactors = F)

View(df)

#############################################################################################
#Data understanding
#############################################################################################
#Checking the dimensions of both data-sets
dim(df)

#Check for NAs
sum(is.na(df))
#There are 41296 NA values the data-set. Let's check number of NAs by attribute.

na.counts <- sapply(df, function(x) sum(is.na(x)))
na.counts <- sort(na.counts,decreasing = T)
View (na.counts)
#All NA values are coming from the postal code column. We will decide what to do with the NA
#values after some EDA

#checking for duplicated rows/values in the data-set
unique.counts <- sapply(df, function(x) length(unique(x)))
unique.counts <- sort(unique.counts,decreasing = T)
View (unique.counts)
#Number of unique values in the "Row.ID" column matches the number of rows in the data-set
#Thus it doesn't look any entries are duplicated.

#Checking the basic summary and structure.
summary(df)
str(df)

#Let's check if there any upper-case vs lower-case mismatches in the Market and
#Segment attributes
summary(as.factor(df$Market))
summary(as.factor(df$Segment))
#There are no issues pertaining to lower-case/upper-case mismatches for any Market
#or Segment category

#############################################################################################
#Data preparation and EDA
#############################################################################################

#Let's get an idea about the distributions of Sales, Quantity, Profit, 
#and Shipping Costs attributes
plot_grid(ggplot(df, aes(Sales))+ geom_histogram(binwidth = 100)+xlim(0,2000), 
          ggplot(df, aes(Profit))+ geom_histogram(binwidth = 10)+xlim(-250,250),
          ggplot(df, aes(Quantity))+ geom_histogram(binwidth = 1),
          align = "h")
#The Sales and Quantity are always positive as expected, whereas profit
#ranges from negative through positive values which is expected from a practical 
#perspective

#Converting the Date columns to date-type
df$Order.Date <- as.POSIXct(df$Order.Date,format="%d-%m-%Y")
#df$Ship.Date <- as.POSIXct(df$Ship.Date,format="%d-%m-%Y")
df$Month <- as.numeric(format(df$Order.Date,"%m"))
df$Year <- as.numeric(format(df$Order.Date,"%Y"))
df$Month <- df$Month+12*(df$Year-min(df$Year))

df_grouped<-df %>% 
  group_by(Market,Segment,Year,Month) %>% 
  summarise(Sales=sum(Sales),Quantity=sum(Quantity),Profit=sum(Profit))

df_profits<-df_grouped %>%
  group_by(Market,Segment) %>%
  summarise(Median_profit=median(Profit),Mean_profit=mean(Profit),
            ProfitCV=(sd(Profit)/mean(Profit)))
#APAC consumer and EU consumer are the two most profitable and consistently profitable buckets.
# This can be seen if we sort the data by profitcv

df_profits <- df_profits[order(-df_profits$Median_profit),]
head(df_profits,2)
#APAC Consumer and EU Consumer buckets have the two highest median profits

df_profits <- df_profits[order(-df_profits$Mean_profit),]
head(df_profits,2)
#Same market buckets as above show the two highest mean profits as well

df_profits <- df_profits[order(df_profits$ProfitCV),]
head(df_profits,2)
#EU Consumer and APAC Consumer buckets show the two lowest ProfitCV values.

#From above analysis APAC and EU Consumer buckets are the most profitable,
#and consistently profitable segments. Thus, further analysis will focus on
#the above buckets.

APAC_consumer <- data.frame(subset(df_grouped,Market=="APAC"&Segment=="Consumer"))
EU_consumer <- data.frame(subset(df_grouped,Market=="EU"&Segment=="Consumer"))

#Let's check how the sales and quantity attributes are varying across the given time period
#for both APAC and EU consumer buckets
plot_grid(ggplot(APAC_consumer, aes(Month,Sales,group=factor(Year),col=factor(Year)))+ geom_line()+labs(title="APAC_sales"), 
          ggplot(APAC_consumer, aes(Month,Quantity,group=factor(Year),col=factor(Year)))+ geom_line()+labs(title="APAC_quantity"),
          ggplot(EU_consumer, aes(Month,Sales,group=factor(Year),col=factor(Year)))+ geom_line()+labs(title="EU_sales"),
          ggplot(EU_consumer, aes(Month,Quantity,group=factor(Year),col=factor(Year)))+ geom_line()+labs(title="EU_quantity"),
          align = "h")
#Both Sales and Quantity seem be to showing a trend and seasonal behavior 
#in the two market buckets.

APAC_sales_total <- ts(APAC_consumer$Sales)
APAC_quantity_total <- ts(APAC_consumer$Quantity)
EU_sales_total <- ts(EU_consumer$Sales)
EU_quantity_total <- ts(EU_consumer$Quantity)

acf(APAC_sales_total)
acf(APAC_quantity_total)
acf(EU_sales_total)
acf(EU_quantity_total)

###########################################################################################
#Modeling - Classical decomposition
###########################################################################################

APAC_sales_ts <- ts(APAC_consumer$Sales[1:42])
APAC_quantity_ts <- ts(APAC_consumer$Quantity[1:42])
EU_sales_ts <- ts(EU_consumer$Sales[1:42])
EU_quantity_ts <- ts(EU_consumer$Quantity[1:42])

Month <- APAC_consumer$Month[1:42]
w <- 1

###################################################
#Outdata and Month values for APAC Consumer bucket
outdata <- APAC_consumer[43:48,]
timevals_out <- outdata$Month
###################################################

##########################################################################
#APAC_Sales
##########################################################################

#Let's smooth the time-series before modeling

APAC_sales_smth <- stats::filter(APAC_sales_ts, 
                         filter=rep(1/(w+1),(w+1)), 
                         method='convolution', sides=2)

APAC_sales_smth[1] <- APAC_sales_ts[1]
APAC_sales_smth[length(APAC_sales_ts)] <- APAC_sales_ts[length(APAC_sales_ts)]

plot(APAC_sales_ts,lw=2)
lines(APAC_sales_smth,col="blue",lw=2)
legend("topleft", c("Raw","Smoothed"), col=c("black","blue"), lwd=2)
#The smoothing seems reasonable without loss of key features of the time series.

#Let's bind the Month vector and APAC_sales_smth time series to create a data-frame,
#which will be used for modeling the trend and seasonality
APAC_sales_smth_df <- as.data.frame(cbind(Month=Month,as.vector(APAC_sales_smth)))
colnames(APAC_sales_smth_df)[2] <- 'Sales'

#Since the Sales and Quantity vs time period plots were showing a drop in the respective
#attribute in the month of Januray for each year, let's consider a modulo function
#which gives a period of 12. Since the peak occurs around the end of each year,
#(Month-1)%%12 will be the function considered. Additionally, there seems to be some
#within year seasonality as well, which will be modeled using sin and cos functions.
lmfit <- lm(Sales ~ sin(Month)*(Month-1)%%12+
              cos(Month)*(Month-1)%%12+
              Month,data=APAC_sales_smth_df)
summary(lmfit)

global_pred <- predict(lmfit, Month=Month)

#Visual check to see how good the trend and seasonality fit is
plot(APAC_sales_smth,lw=2)
lines(global_pred,col='red',lw=2)
legend("topleft", c("Smoothed","Fitted"), col=c("black","red"), lwd=2)
#The global_pred fit seems reasonable.

#Let's see what the locally variable time-series looks like, and see if we can
#model it using ARIMA modeling.
local_pred=APAC_sales_smth - global_pred
plot(local_pred,lw=2)
acf(local_pred)
acf(local_pred,type='partial')

armafit <- auto.arima(local_pred)
tsdiag(armafit)
armafit
#We are getting a ARIMA(0,0,1) model or equivalently a MA(1) model

#Let's check the residuals to see if it's white noise or not
resi <- local_pred-fitted(armafit)
plot(resi,lw=2)
#The residuals seem random enough, however let's do the adf and kpss tests to confirm if it's
#indeed white noise
adf.test(local_pred,alternative = "stationary")
kpss.test(local_pred)
#Based on above tests, the residuals is most likely white noise.

#Let's predict the Sales for next 6 months and see what kind of accuracy we get
#Forecast for auto-regressive part
arma_fcast <- predict(armafit, n.ahead = 6)

#Forecast for globally predictive part
lm_fcast <- predict(lmfit,data.frame(Month=timevals_out))

#Adding the auto-regressive and global trend forecasts to get the total forecast
global_pred_out <- arma_fcast$pred+lm_fcast

fcast <- global_pred_out

#Accuracy of forecast vs original time series.
MAPE_class_dec <- accuracy(fcast,outdata[,5])[5]
MAPE_class_dec
#MAPE is ~20%, which seems decent.

#Plotting the fit+prediction vs the original time series to get a visual sense of the goodness
#of fit
class_dec_pred <- c(ts(global_pred),ts(global_pred_out))
plot(APAC_sales_total, col = "black",lw=2,type='b')
lines(class_dec_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

##########################################################################
#APAC_Quantity
##########################################################################

#Smoothing the time-series before modeling
APAC_quantity_smth <- stats::filter(APAC_quantity_ts, 
                                    filter=rep(1/(w+1),(w+1)), 
                                    method='convolution', sides=2)

APAC_quantity_smth[1] <- APAC_quantity_ts[1]
APAC_quantity_smth[length(APAC_quantity_ts)] <- APAC_quantity_ts[length(APAC_quantity_ts)]

plot(APAC_quantity_ts,lw=2)
lines(APAC_quantity_smth,col="blue",lw=2)
legend("topleft", c("Raw","Smoothed"), col=c("black","blue"), lwd=2)
#The smoothing seems reasonable without loss of key features of the time series.

#Let's bind the Month vector and APAC_sales_smth time series to create a data-frame,
#which will be used for modeling the trend and seasonality
APAC_quantity_smth_df <- as.data.frame(cbind(Month=Month,as.vector(APAC_quantity_smth)))
colnames(APAC_quantity_smth_df)[2] <- 'Quantity'

lmfit <- lm(Quantity ~ sin(Month)*(Month-1)%%12+
              cos(Month)*(Month-1)%%12+
              Month,data=APAC_quantity_smth_df)
summary(lmfit)

global_pred <- predict(lmfit, Month=Month)

#Visual check to see how good the trend and seasonality fit is
plot(APAC_quantity_smth,lw=2)
lines(global_pred,col='red',lw=2)
legend("topleft", c("Smoothed","Fitted"), col=c("black","red"), lwd=2)
#The global_pred fit seems reasonable.

#Let's see what the locally variable time-series looks like, and see if we can
#model it using ARIMA modeling.
local_pred=APAC_quantity_smth - global_pred
plot(local_pred,lw=2)
acf(local_pred)
acf(local_pred,type='partial')

armafit <- auto.arima(local_pred)
tsdiag(armafit)
armafit
#We are getting a ARIMA(0,0,1) model or equivalently a MA(1) model

#Let's check the residuals to see if it's white noise or not
resi <- local_pred-fitted(armafit)
plot(resi,lw=2)
#The residuals seem random enough, however let's do the adf and kpss tests to confirm if it's
#indeed white noise
adf.test(local_pred,alternative = "stationary")
kpss.test(local_pred)
#Based on above tests, the residuals is most likely white noise.

#Let's predict the Sales for next 6 months and see what kind of accuracy we get
#Forecast for auto-regressive part
arma_fcast <- predict(armafit, n.ahead = 6)

#Forecast for globally predictive part
lm_fcast <- predict(lmfit,data.frame(Month=timevals_out))

#Adding the auto-regressive and global trend forecasts to get the total forecast
global_pred_out <- arma_fcast$pred+lm_fcast
fcast <- global_pred_out

#Accuracy of forecast vs original time series.
MAPE_class_dec <- accuracy(fcast,outdata[,6])[5]
MAPE_class_dec
#MAPE is ~24%, which seems decent.

#Plotting the fit+prediction vs the original time series to get a visual sense of the goodness
#of fit
class_dec_pred <- c(ts(global_pred),ts(global_pred_out))
plot(APAC_quantity_total, col = "black",lw=2,type='b')
lines(class_dec_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

###################################################
#Outdata and Month values for EU Consumer bucket
outdata <- EU_consumer[43:48,]
timevals_out <- outdata$Month
###################################################

##########################################################################
#EU_Sales
##########################################################################

#Let's smooth the time-series before modeling

EU_sales_smth <- stats::filter(EU_sales_ts, 
                                 filter=rep(1/(w+1),(w+1)), 
                                 method='convolution', sides=2)

EU_sales_smth[1] <- EU_sales_ts[1]
EU_sales_smth[length(EU_sales_ts)] <- EU_sales_ts[length(EU_sales_ts)]

plot(EU_sales_ts,lw=2)
lines(EU_sales_smth,col="blue",lw=2)
legend("topleft", c("Raw","Smoothed"), col=c("black","blue"), lwd=2)
#The smoothing seems reasonable without loss of key features of the time series.

#Let's bind the Month vector and EU_sales_smth time series to create a data-frame,
#which will be used for modeling the trend and seasonality
EU_sales_smth_df <- as.data.frame(cbind(Month=Month,as.vector(EU_sales_smth)))
colnames(EU_sales_smth_df)[2] <- 'Sales'

lmfit <- lm(Sales ~ sin(1.16*Month)*(Month-1)%%12+cos(1.219*Month)*(Month-1)%%12+
              Month,data=EU_sales_smth_df)
summary(lmfit)

global_pred <- predict(lmfit, Month=Month)

#Visual check to see how good the trend and seasonality fit is
plot(EU_sales_smth,lw=2)
lines(global_pred,col='red',lw=2)
legend("topleft", c("Smoothed","Fitted"), col=c("black","red"), lwd=2)
#The global_pred fit seems reasonable.

#Let's see what the locally variable time-series looks like, and see if we can
#model it using ARIMA modeling.
local_pred=EU_sales_smth - global_pred
plot(local_pred,lw=2)
acf(local_pred)
acf(local_pred,type='partial')

armafit <- auto.arima(local_pred)
tsdiag(armafit)
armafit
#We are getting a ARIMA(0,0,0) model

#Let's check the residuals to see if it's white noise or not
resi <- local_pred-fitted(armafit)
plot(resi,lw=2)
#The residuals seem random enough, however let's do the adf and kpss tests to confirm if it's
#indeed white noise
adf.test(local_pred,alternative = "stationary")
kpss.test(local_pred)
#Based on above tests, the residuals is most likely white noise.

#Let's predict the Sales for next 6 months and see what kind of accuracy we get
#Forecast for auto-regressive part
arma_fcast <- predict(armafit, n.ahead = 6)

#Forecast for globally predictive part
lm_fcast <- predict(lmfit,data.frame(Month=timevals_out))

#Adding the auto-regressive and global trend forecasts to get the total forecast
global_pred_out <- arma_fcast$pred+lm_fcast

fcast <- global_pred_out

#Accuracy of forecast vs original time series.
MAPE_class_dec <- accuracy(fcast,outdata[,5])[5]
MAPE_class_dec
#MAPE is ~23%, which seems decent.

#Plotting the fit+prediction vs the original time series to get a visual sense of the goodness
#of fit
class_dec_pred <- c(ts(global_pred),ts(global_pred_out))
plot(EU_sales_total, col = "black",lw=2,type='b')
lines(class_dec_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

##########################################################################
#EU_Quantity
##########################################################################

#Smoothing the time-series before modeling
EU_quantity_smth <- stats::filter(EU_quantity_ts, 
                                    filter=rep(1/(w+1),(w+1)), 
                                    method='convolution', sides=2)

EU_quantity_smth[1] <- EU_quantity_ts[1]
EU_quantity_smth[length(EU_quantity_ts)] <- EU_quantity_ts[length(EU_quantity_ts)]

plot(EU_quantity_ts,lw=2)
lines(EU_quantity_smth,col="blue",lw=2)
legend("topleft", c("Raw","Smoothed"), col=c("black","blue"), lwd=2)
#The smoothing seems reasonable without loss of key features of the time series.

#Let's bind the Month vector and EU_sales_smth time series to create a data-frame,
#which will be used for modeling the trend and seasonality
EU_quantity_smth_df <- as.data.frame(cbind(Month=Month,as.vector(EU_quantity_smth)))
colnames(EU_quantity_smth_df)[2] <- 'Quantity'

lmfit <- lm(Quantity ~ sin(1.16*Month)*(Month-1)%%12+cos(1.227*Month)*(Month-1)%%12+
              Month,data=EU_quantity_smth_df)
summary(lmfit)

global_pred <- predict(lmfit, Month=Month)

#Visual check to see how good the trend and seasonality fit is
plot(EU_quantity_smth,lw=2)
lines(global_pred,col='red',lw=2)
legend("topleft", c("Smoothed","Fitted"), col=c("black","red"), lwd=2)
#The global_pred fit seems reasonable.

#Let's see what the locally variable time-series looks like, and see if we can
#model it using ARIMA modeling.
local_pred=EU_quantity_smth - global_pred
plot(local_pred,lw=2)
acf(local_pred)
acf(local_pred,type='partial')

armafit <- auto.arima(local_pred)
tsdiag(armafit)
armafit
#We are getting a ARIMA(0,0,0) model or equivalently a MA(1) model

#Let's check the residuals to see if it's white noise or not
resi <- local_pred-fitted(armafit)
plot(resi,lw=2)
#The residuals seem random enough, however let's do the adf and kpss tests to confirm if it's
#indeed white noise
adf.test(local_pred,alternative = "stationary")
kpss.test(local_pred)
#Based on above tests, the residuals is most likely white noise.

#Let's predict the Sales for next 6 months and see what kind of accuracy we get
#Forecast for auto-regressive part
arma_fcast <- predict(armafit, n.ahead = 6)

#Forecast for globally predictive part
lm_fcast <- predict(lmfit,data.frame(Month=timevals_out))

#Adding the auto-regressive and global trend forecasts to get the total forecast
global_pred_out <- arma_fcast$pred+lm_fcast
fcast <- global_pred_out

#Accuracy of forecast vs original time series.
MAPE_class_dec <- accuracy(fcast,outdata[,6])[5]
MAPE_class_dec
#MAPE is ~26%, which seems decent.

#Plotting the fit+prediction vs the original time series to get a visual sense of the goodness
#of fit
class_dec_pred <- c(ts(global_pred),ts(global_pred_out))
plot(EU_quantity_total, col = "black",lw=2,type='b')
lines(class_dec_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

#Thus the MAPE value for the respective Market and Segment using 
#Classical decomposition is as follows:
#APAC_Sales: ~20%
#APAC_Quantity: ~24%
#EU_Sales: ~23%
#EU_Quantity: ~26%

###########################################################################################
#Modeling - Auto-ARIMA
###########################################################################################

###################################################
#Outdata and Month values for APAC Consumer bucket
outdata <- APAC_consumer[43:48,]
###################################################

##########################################################################
#APAC_Sales
##########################################################################

autoarima <- auto.arima(APAC_sales_ts)
autoarima
plot(autoarima$x, col="black",lw=2)
lines(fitted(autoarima), col="red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

#Again, let's check if the residual series is white noise
resi_auto_arima <- APAC_sales_ts - fitted(autoarima)

adf.test(resi_auto_arima,alternative = "stationary")
kpss.test(resi_auto_arima)

#Also, let's evaluate the model using MAPE
fcast_auto_arima <- predict(autoarima, n.ahead = 6)

MAPE_auto_arima <- accuracy(fcast_auto_arima$pred,outdata$Sales)[5]
MAPE_auto_arima
#27.68952

#Lastly, let's plot the predictions along with original values, to
#get a visual feel of the fit
auto_arima_pred <- c(fitted(autoarima),ts(fcast_auto_arima$pred))
plot(APAC_sales_total, col = "black",lw=2,type='b')
lines(auto_arima_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

##########################################################################
#APAC_Quantity
##########################################################################

autoarima <- auto.arima(APAC_quantity_ts)
autoarima
plot(autoarima$x, col="black",lw=2)
lines(fitted(autoarima), col="red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

#Again, let's check if the residual series is white noise
resi_auto_arima <- APAC_quantity_ts - fitted(autoarima)

adf.test(resi_auto_arima,alternative = "stationary")
kpss.test(resi_auto_arima)

#Also, let's evaluate the model using MAPE
fcast_auto_arima <- predict(autoarima, n.ahead = 6)

MAPE_auto_arima <- accuracy(fcast_auto_arima$pred,outdata$Quantity)[5]
MAPE_auto_arima
#26.24458

#Lastly, let's plot the predictions along with original values, to
#get a visual feel of the fit
auto_arima_pred <- c(fitted(autoarima),ts(fcast_auto_arima$pred))
plot(APAC_quantity_total, col = "black",lw=2,type='b')
lines(auto_arima_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

###################################################
#Outdata and Month values for APAC Consumer bucket
outdata <- EU_consumer[43:48,]
###################################################

##########################################################################
#EU_Sales
##########################################################################

autoarima <- auto.arima(EU_sales_ts)
autoarima
plot(autoarima$x, col="black",lw=2)
lines(fitted(autoarima), col="red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

#Again, let's check if the residual series is white noise
resi_auto_arima <- EU_sales_ts - fitted(autoarima)

adf.test(resi_auto_arima,alternative = "stationary")
kpss.test(resi_auto_arima)

#Also, let's evaluate the model using MAPE
fcast_auto_arima <- predict(autoarima, n.ahead = 6)

MAPE_auto_arima <- accuracy(fcast_auto_arima$pred,outdata$Sales)[5]
MAPE_auto_arima
#28.9226

#Lastly, let's plot the predictions along with original values, to
#get a visual feel of the fit
auto_arima_pred <- c(fitted(autoarima),ts(fcast_auto_arima$pred))
plot(EU_sales_total, col = "black",lw=2,type='b')
lines(auto_arima_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

##########################################################################
#EU_Quantity
##########################################################################

autoarima <- auto.arima(EU_quantity_ts)
autoarima
plot(autoarima$x, col="black",lw=2)
lines(fitted(autoarima), col="red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

#Again, let's check if the residual series is white noise
resi_auto_arima <- EU_quantity_ts - fitted(autoarima)

adf.test(resi_auto_arima,alternative = "stationary")
kpss.test(resi_auto_arima)

#Also, let's evaluate the model using MAPE
fcast_auto_arima <- predict(autoarima, n.ahead = 6)

MAPE_auto_arima <- accuracy(fcast_auto_arima$pred,outdata$Quantity)[5]
MAPE_auto_arima
#30.13319

#Lastly, let's plot the predictions along with original values, to
#get a visual feel of the fit
auto_arima_pred <- c(fitted(autoarima),ts(fcast_auto_arima$pred))
plot(EU_quantity_total, col = "black",lw=2,type='b')
lines(auto_arima_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted"), col=c("black","red"), lwd=2)

#Thus the MAPE value for the respective Market and Segment using 
#Auto-ARIMA is as follows:
#APAC_Sales: ~28%
#APAC_Quantity: ~27%
#EU_Sales: ~29%
#EU_Quantity: ~30%

#Thus for all four buckets, Classical decomposition is giving a better MAPE value compared
#to the Auto-ARIMA model. It also seems that seasonality for the EU Market is slightly
#different compared to APAC. EU market is perhaps showing a slight quarterly
#seasonality on top of yearly seasonality, based on ACF plots and the Sales and Quantity vs
#Month plots. Perhaps the modulo function along-with sin and cos can be tweaked to give
#better MAPE values for the EU market.

###########################################################################################
#Forecast for months from 49 through 54
###########################################################################################

#Since Classical decomposition is showing better MAPE values, we will be using those
#models for predicting Sales and Quantity from months 49 through 54 for the four market
#buckets

Fcast_Month <- c(43:54)

##########################################################################
#APAC_Sales
##########################################################################

lmfit <- lm(Sales ~ sin(Month)*(Month-1)%%12+
              cos(Month)*(Month-1)%%12+
              Month,data=APAC_sales_smth_df)

global_pred <- predict(lmfit, Month=Month)

local_pred=APAC_sales_smth - global_pred

armafit <- auto.arima(local_pred)

#Predicting Sales (locally predictable part) for next 12 months in order 
#to get the forecast for months 49 through 54
arma_fcast <- predict(armafit, n.ahead = 12)

#Forecast for globally predictive part
lm_fcast <- predict(lmfit,data.frame(Month=Fcast_Month))

#Adding the auto-regressive and global trend forecasts to get the total forecast
global_pred_out <- arma_fcast$pred+lm_fcast

fcast <- global_pred_out

#Plotting the fit+forecast vs the original time series 
class_dec_pred <- c(ts(global_pred),ts(global_pred_out))
plot(APAC_sales_total, col = "black",lw=2,type='b',xlim=c(0,54))
lines(class_dec_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted+Forecast"), col=c("black","red"), lwd=2)

##########################################################################
#APAC_Quantity
##########################################################################

lmfit <- lm(Quantity ~ sin(Month)*(Month-1)%%12+
              cos(Month)*(Month-1)%%12+
              Month,data=APAC_quantity_smth_df)

global_pred <- predict(lmfit, Month=Month)

local_pred=APAC_quantity_smth - global_pred

armafit <- auto.arima(local_pred)

#Predicting Sales (locally predictable part) for next 12 months in order 
#to get the forecast for months 49 through 54
arma_fcast <- predict(armafit, n.ahead = 12)

#Forecast for globally predictive part
lm_fcast <- predict(lmfit,data.frame(Month=Fcast_Month))

#Adding the auto-regressive and global trend forecasts to get the total forecast
global_pred_out <- arma_fcast$pred+lm_fcast

fcast <- global_pred_out

#Plotting the fit+forecast vs the original time series 
class_dec_pred <- c(ts(global_pred),ts(global_pred_out))
plot(APAC_quantity_total, col = "black",lw=2,type='b',xlim=c(0,54))
lines(class_dec_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted+Forecast"), col=c("black","red"), lwd=2)

##########################################################################
#EU_Sales
##########################################################################

lmfit <- lm(Sales ~ sin(1.16*Month)*(Month-1)%%12+cos(1.219*Month)*(Month-1)%%12+
              Month,data=EU_sales_smth_df)

global_pred <- predict(lmfit, Month=Month)

local_pred=EU_sales_smth - global_pred

armafit <- auto.arima(local_pred)

#Predicting Sales (locally predictable part) for next 12 months in order 
#to get the forecast for months 49 through 54
arma_fcast <- predict(armafit, n.ahead = 12)

#Forecast for globally predictive part
lm_fcast <- predict(lmfit,data.frame(Month=Fcast_Month))

#Adding the auto-regressive and global trend forecasts to get the total forecast
global_pred_out <- arma_fcast$pred+lm_fcast

fcast <- global_pred_out

#Plotting the fit+forecast vs the original time series 
class_dec_pred <- c(ts(global_pred),ts(global_pred_out))
plot(EU_sales_total, col = "black",lw=2,type='b',xlim=c(0,54))
lines(class_dec_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted+Forecast"), col=c("black","red"), lwd=2)

##########################################################################
#EU_Quantity
##########################################################################

lmfit <- lm(Quantity ~ sin(1.16*Month)*(Month-1)%%12+cos(1.227*Month)*(Month-1)%%12+
              Month,data=EU_quantity_smth_df)

global_pred <- predict(lmfit, Month=Month)

local_pred=EU_quantity_smth - global_pred

armafit <- auto.arima(local_pred)

#Predicting Sales (locally predictable part) for next 12 months in order 
#to get the forecast for months 49 through 54
arma_fcast <- predict(armafit, n.ahead = 12)

#Forecast for globally predictive part
lm_fcast <- predict(lmfit,data.frame(Month=Fcast_Month))

#Adding the auto-regressive and global trend forecasts to get the total forecast
global_pred_out <- arma_fcast$pred+lm_fcast

fcast <- global_pred_out

#Plotting the fit+forecast vs the original time series 
class_dec_pred <- c(ts(global_pred),ts(global_pred_out))
plot(EU_quantity_total, col = "black",lw=2,type='b',xlim=c(0,54))
lines(class_dec_pred, col = "red",lw=2)
legend("topleft", c("Raw","Fitted+Forecast"), col=c("black","red"), lwd=2)
