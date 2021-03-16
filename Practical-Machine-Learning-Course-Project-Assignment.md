## Predictions using the Weight Lifting Exercise Dataset

## 1 - Summary

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement ??? a group of enthusiasts who take
measurements about themselves regularly to improve their health, to find
patterns in their behavior, or because they are tech geeks.

One thing that people regularly do is quantify how much of a particular
activity they do, but they rarely quantify how well they do it. In this
project, your goal will be to use data from accelerometers on the belt,
forearm, arm, and dumbbell of 6 participants. They were asked to perform
barbell lifts correctly and incorrectly in 5 different ways. More
information is available from the website here:
<http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight
Lifting Exercise Dataset).

This project has the purpose to predict the manner in which users
perform the exercises. There are 5 possible results, reported in the
`classe` variable:

-   A: exactly according to the specification
-   B: throwing the elbows to the front
-   C: lifting the dumbbell only halfway
-   D: lowering the dumbbell only halfway
-   E: throwing the hips to the front

The objective of this project is to predict the `classe` based on data
from accelerometers on the belt, forearm, arm, and dumbbell of 6
participants.

## 2 - Libraries

    library(knitr)
    knitr::opts_chunk$set(echo = TRUE, warning=FALSE, message=FALSE, fig.width=10, fig.height=5)
    options(width=120)
    library(lattice); library(ggplot2); library(plyr)
    library(caret); library(randomForest); library(rpart);library(rpart.plot); library(tree)

    ## Warning: package 'caret' was built under R version 4.0.4

    ## Warning: package 'randomForest' was built under R version 4.0.4

    ## randomForest 4.6-14

    ## Type rfNews() to see new features/changes/bug fixes.

    ## 
    ## Attaching package: 'randomForest'

    ## The following object is masked from 'package:ggplot2':
    ## 
    ##     margin

    ## Warning: package 'rpart.plot' was built under R version 4.0.4

    ## Warning: package 'tree' was built under R version 4.0.4

    library(rattle)

    ## Warning: package 'rattle' was built under R version 4.0.4

    ## Loading required package: tibble

    ## Loading required package: bitops

    ## Rattle: A free graphical interface for data science with R.
    ## Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    ## Type 'rattle()' to shake, rattle, and roll your data.

    ## 
    ## Attaching package: 'rattle'

    ## The following object is masked from 'package:randomForest':
    ## 
    ##     importance

    set.seed(6266) # set contact random seed

## 3 - Dataset Loading

## 4 - Wrangling Dataset

Organizing a new dataset with only the data that is necessary for EDA
and ML Training data.

    ## Reading dataset and replacing NA Strings with NA
    training <- read.csv("pml-training.csv", na.strings=c("NA","#DIV/0!",""))
    testing <- read.csv("pml-testing.csv", na.strings=c("NA","#DIV/0!",""))
    dim(training); dim(testing)

    ## [1] 19622   160

    ## [1]  20 160

    ##  Some variables (7 first columns with 'X', 'user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 
    ##  'cvtd_timestamp', 'new_window', 'num_window') can be deleted because will not useful to this Project.
    training <- training[,-c(1:7)]
    testing <- testing[,-c(1:7)]
    dim(training); dim(testing)

    ## [1] 19622   153

    ## [1]  20 153

    # Delete columns with all missing values
    training <- training[ , colSums(is.na(training)) == 0]
    testing <- testing[ , colSums(is.na(testing)) == 0]
    dim(training); dim(testing)

    ## [1] 19622    53

    ## [1] 20 53

With our cleanup, as you can see, now we have a reduced number of
variables, that will be used for analysis.

## 5 - Plotting Dataset for EDA

This project has purpose to predict the manner in which users perform
the exercises based on variable `classe` classified as A,B,C,D and E.

Lets use the variable `classe` in the plot, to see the frequency of each
levels in training dataset.

The A classe seems to be the most frequent and D classe the least.

## 6 - Partition Training Dataset

To enable cross-validation lets make two subsets using 70% for Training
and 30% for Testing. This method will allow us to apply ML Prediction
Models and evaluate bias and variance.

    partition <- createDataPartition(y=training$classe, p=0.7, list=FALSE)
    trainingSet <- training[partition, ] 
    testingSet <- training[-partition, ]
    rm(partition)

## 7 - Predictions Models

Lets implement the Regression Tree model and Random Forest.

### 7.1 - Regression Tree

Decision tree model is good for classification problems.

### 7.2 - RPart from the Caret package.

    rpartFit <- train(classe ~ .,method="rpart",data=trainingSet) #library Caret
    print(rpartFit$finalModel)

    ## n= 13737 
    ## 
    ## node), split, n, loss, yval, (yprob)
    ##       * denotes terminal node
    ## 
    ##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
    ##    2) roll_belt< 130.5 12585 8687 A (0.31 0.21 0.19 0.18 0.11)  
    ##      4) pitch_forearm< -33.95 1103    9 A (0.99 0.0082 0 0 0) *
    ##      5) pitch_forearm>=-33.95 11482 8678 A (0.24 0.23 0.21 0.2 0.12)  
    ##       10) magnet_dumbbell_y< 439.5 9707 6960 A (0.28 0.18 0.24 0.19 0.11)  
    ##         20) roll_forearm< 122.5 5995 3538 A (0.41 0.18 0.18 0.17 0.06) *
    ##         21) roll_forearm>=122.5 3712 2501 C (0.078 0.18 0.33 0.23 0.18) *
    ##       11) magnet_dumbbell_y>=439.5 1775  874 B (0.032 0.51 0.044 0.22 0.2) *
    ##    3) roll_belt>=130.5 1152    8 E (0.0069 0 0 0 0.99) *

    fancyRpartPlot(rpartFit$finalModel) #library Rattle

![](Practical-Machine-Learning-Course-Project-Assignment_files/figure-markdown_strict/unnamed-chunk-6-1.png)

Rpart has a little much accurate result than Tree. Rpart has some
methods, and one of them is `class` with good precision. Lets try it.
te. See below.

    classFit <- rpart(classe ~ ., data=trainingSet, method="class") #library caret
    fancyRpartPlot(classFit, sub = "") #library Rattle

![](Practical-Machine-Learning-Course-Project-Assignment_files/figure-markdown_strict/unnamed-chunk-7-1.png)

Now,lets see how well it is predicting our test data:

The accuracy shows a good results using Rpart, but a great Regression
Method, used for resolution of problems in Kaggle, is Random Forests.
Lets apply it now into our Project.

### 7.3 - Random Forest

As seen by the result of the confusionmatrix, the accuracy for random
forests was 0.9958 (very good sensitivity and specificity values)
whereas the decision tree was 0.69. The accuracy is above 99% for the
random forest model in our cross-validation data with few
misclassifications as compared to the decision tree model.

## 8 - Submission Data for Grading

### 8.1 - Trained Model on the Twenty Testing Data

Random Forests was the choose for this project. Lets apply it to our
Testing dataset using the predictor on the test data.

The answer above is the model machine learning algorithm applied to the
20 test cases available in the test data. It scores 100% of the
submission (the 20 values to be predict), that will be submitted to
answer the questions of Course Project Prediction Quiz.
