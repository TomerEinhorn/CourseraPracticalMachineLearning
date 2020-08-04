---
title: "Practical Machine Learning - Course Project"
author: "Tomer Einhorn"
date: "8/4/2020"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE,fig.width=12, fig.height=8, fig.path='Figs/',
                      warning=FALSE, message=FALSE, comment=NA)
```

## Overview

This document is given as part of the final project in Coursera's Practical Machine Learning Course, which is part of the Data Science Specialization. 
The main goal of this project is to predict the manner in which several participant in an experiment perform certain physical exercises as described in the following sections. In terms of the given data sets, the aim of this project is to predict the values of the "classe" variable in the given train set. The machine learning akgorithm described in this document is applied to the given 20 test cases to see how accurate it is. 

## Introduction

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data Loading and Cleaning

### DataSet Overview

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

## Libraries needed and other setup

```{r}
set.seed(123456789)
library(caret) 
library(rpart)
library(rattle) # needed to plot fancy decision tree
```


## Data uploading


```{r}
trainUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"

testUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

download.file(trainUrl, destfile= "./training.csv")
download.file(testUrl, destfile= "./test.csv")
training <- read.csv("training.csv")
testing <- read.csv("test.csv")

inTrain <- createDataPartition(y = training$classe, p=0.6, list=FALSE)
train <- training[inTrain,]
test <- training[-inTrain,]

dim(train)
dim(test)

```

As we can see, both the train and test data sets contain 160 variables (columns). 
But, as can be seen by the result of running the following code, a lot of the columns in both the train set and the test set contain a lot of NA values:

```{r}
trainNAs <- sapply(train, function(x) mean(is.na(x)))
testNAs <- sapply(test, function(x) mean(is.na(x)))
table(trainNAs)
table(testNAs)

```

Therefore, we'll remove from both data sets the columns that contain that little real data:

```{r}
train <- train[, -trainNAs==FALSE]
test <- test[, -testNAs==FALSE]

```

Now, we have a lot less variables:

```{r}
dim(train)
dim(test)

```

We'll also attempt to remove all columns with zero or near zero variation, as they explain little (or non at all) variance in the "classe" variable value.

```{r}
nsv <- (nearZeroVar(train))
train <- train[,-nsv]
test <- test[, -nsv]
dim(train)
dim(test)

```

We'll also remove the first 6 columns as they are related to identification only:

```{r}
train <- train[,-(1:6)]
test <- test[, -(1:6)]
dim(train)
dim(test)

```

By this cleaning process we were able to reduce the number of columns in the training data set from 160 to 53, without losing a lot of valuable information, 

## Prediction Models

We will use three predicting models to the train data set, and choose the best performing (in terms of accuracy) to predict the quiz results (the test set).
The three models are:
1. Random Forest.
2. Decision Tree.
3. Generalized boosted Regression model.
A confusion matrix is plotted for each of the models to visualize each of the models accuracy.

### 1. Random Forest

#### Model Fitting

```{r}
ControlRandomForest <- trainControl(method="cv", number=3, verboseIter=FALSE)
modFitRF <- train(classe ~ .,data=train,method="rf",trControl=ControlRandomForest)
modFitRF$finalModel

```

#### Prediction on test set 

```{r}
RFpredict <- predict(modFitRF, newdata = test)
RFpredictConfMat <- confusionMatrix(factor(RFpredict), factor(test$classe))
RFpredictConfMat
```

Let's see how accurate this model is:

```{r}
RFpredictConfMat$overall['Accuracy']
```
<span style="color: red;"><b> So the Random Forest Model results an accuracy of 0.9913!</b> </span>

#### Plotting matrix results
```{r}
plot(RFpredictConfMat$table, col = RFpredictConfMat$byClass, main = "Random Forest Prediction Matrix")

```

### 2. Decision Tree

#### Model Fitting

```{r}
modFitDT <- rpart(classe ~ .,data=train,method="class")
fancyRpartPlot(modFitDT)
```

#### Prediction on testset 

```{r}
DTpredict <- predict(modFitDT, newdata = test, type = "class")
DTpredictConfMat <- confusionMatrix(factor(DTpredict), factor(test$classe))
DTpredictConfMat

```

Let's see how accurate this model is:

```{r}
DTpredictConfMat$overall['Accuracy']
```
<span style="color: red;"><b> So the Random Forest Model results an accuracy of 0.6969!</b> </span>

#### Plotting matrix results

```{r}
plot(DTpredictConfMat$table, col = DTpredictConfMat$byClass, main = "Decision Tree Prediction Matrix")

```

### 3. Generalized Boosted Regression Model 

#### Model Fitting

```{r}
ControlGBM <- trainControl(method="repeatedcv", number=6, repeats = 1)
modFitGBM <- train(classe ~ .,data=train,method="gbm", trControl=ControlGBM,
                   verbose=FALSE)
modFitGBM$finalModel

```

#### Prediction on testset 

```{r}
GBMpredict <- predict(modFitGBM, newdata = test)
GBMpredictConfMat <- confusionMatrix(factor(GBMpredict), factor(test$classe))
GBMpredictConfMat

```

Let's see how accurate this model is:

```{r}
GBMpredictConfMat$overall['Accuracy']
```
<span style="color: red;"><b> So the Random Forest Model results an accuracy of 0.9591!</b> </span>

#### Plotting matrix results

```{r}
plot(GBMpredictConfMat$table, col = GBMpredictConfMat$byClass, 
     main = "Generalized Boosted Model Prediction Matrix")

```

## Final Model Choice and Prediction on Test Set:

To sum up, let's review the accuracy of the 3 models above:

1. Random Forest: 0.9913.
2. Decision Tree: 0.6969.
3. Generalized Boosted Regression Models: 0.9591

As we can see, Random Forest is the most accurate model out of the three, and therefore it will be applied to predict the 20 observations in the test set (and the quiz):

```{r}
testingprediction <- predict(modFitRF, newdata=testing)
testingprediction
```
