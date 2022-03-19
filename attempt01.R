##clear
rm(list=ls())

##imports


##set directory
setwd("C:/Users/New/Documents/kaggle/house-prices_advanced-regression-techniques")

##load data

df_train <- read.csv("train.csv", header=T, sep = ",")

##check the data
View(df_train)

