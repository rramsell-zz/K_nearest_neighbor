# K_nearest_neighbor

Classification Analysis

Proposal of Question

Question

What is the accuracy of the classification method which uses k-nearest neighbor calculations in predicting the target variable churn?

Defined Goal

Build a logistic model to predict the variable churn using the classification method k-nearest neighbor; then, determine the accuracy of the model.

Explanation of Classification Method

The k-nearest neighbor classification method allows for a moving area under curve (AUC). This allows for a flexible approach to the data set. How the method works is by determining those data points closes in space ‘k’ to the input and classifying the data to a similar category.

In example, considering the following array. If points one through ten have corresponding y-values under ten, and points eleven through twenty have y-values over ten and an input of x (or predictor variable) is provided and is the k-nearest neighbor of values under ten, the target variable predicted is one through ten. Thus, the classification method is put into practice within the supervised learning model.

Summary of Method Assumption

There are a few assumptions of this classification model. K-nearest neighbor assumes categorical and continuous x-inputs. The output of the classification method is binary in nature resulting in logistical model prediction. Furthermore, this method works well with highly correlated predictor to target relationships. Reason being, the higher each predictor is correlated, the more accurate the nearest points are similar in their classification.

Programming Language

Python

Packages Used

import pandas as pd 
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
import matplotlib.pyplot as plt
%matplotlib inline
import base64
from IPython.display import HTML

Justification of Programming Language and Packages

The above language and packages are essential to performing a logistic regression model. Python was chosen as the preferred language over R as it offers more versatility in manipulating the train and test process of the model. Pandas was used in the data preparation phase. It allows for effective transformations within the dataset. The sklearn packages offer versatility in sample size, ease of k-nearest neighbor classifications and accuracy tests. The stats models package offers efficiency in fitting trains of data to a model. Python once again, is preferable to R because of the fact it has so many useful packages allowing versatility in programming. 

Data Processing

Imputation will be a preprocessing goal performed on the data set in order to improve the accuracy of the model and classification method. With the package imputation from sklearn, the data may be cleansed from null entries such as: zeros, null values, improper entries, and nonsensical entries (like negative time entries for outage seconds per week).

Steps for Analysis

Below are the steps used to prepare the dataset for k-nearest neighbor classification and logistical model fitting. 

1.	Import packages and data set

2. Clean and Transform the Data

4.	Perform a Train/Test Split

5.	Use K-Nearest Neighbor Classification Method on the Train

6.	Perform Confusion Matrix and Classification Report for Model Testing

7.	Display the Y-Intercepts and Coefficients of the Train

8.	Fit the Trains to a Model and Display the Model Summary


Splitting the Data

The test size was a moderate 80/20 split. This means that twenty percent of the dataset was used for the test train with a random state of five. Eight thousand rows and all nine of the reduced variables from the original data set for the x and y trains. Twenty percent of the original dataset was used with a random state sample of five resulting in two thousand rows for the x and y test sets.

This split is justified by the accuracy of the model. The accuracy being 84%, much higher than the initial model without the split, classification, and reduction.

Output and Intermediate Calculations

The analysis technique used to understand the data for the classification problem and model fitting are described below.

1.	Data Type and Null Count

2.	Data Frame Shape

3.	Train and Test Shapes

4.	Random State and Sample Size for Tests

5.	Model Parameter Display

6.	K-Nearest Neighbor Accuracy Test

7.	Model Accuracy Test

8.	Confusion Matrix

9.	Plot ROC for the Model

10.	Plot ROC for the Classification Method

11.	Determine Area Under Curve Score Arrays for the Classifier and Logistic Model

Accuracy and AUC

Accuracy

The following code was used to determine the accuracy of the classification method. The accuracy was 84%.

Code Used:
classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(x_train, y_train)
y_guess = classifier.predict(x_test)
display(confusion_matrix(y_test, y_guess))
display(classification_report(y_test, y_guess

Area Under Curve (AUC) Score

The k-nearest neighbor classification method has an AUC score array of:

array([0.91859878, 0.90696019, 0.90683228, 0.88695305, 0.91436591,
       0.91660669, 0.89379972, 0.87223254, 0.88663662])
 
Explanation of Scoring

The accuracy score is the likelihood of the classification method correctly predicting the accurate value of the corresponding y-train value. In layman terms, how accurate the prediction is. The k-nearest neighbor classification method with the dataset had an accuracy of 84%, meaning it can predict the target given the predictors correctly 84% of the time.

The AOC score provides further insight into each variable’s accuracy within the classification method. For example, the predictor variable tenure has a 91% accuracy within the array of the classification method. Each of these predictors providing to the overall accuracy score of the technique.
Results and Implications
Results

The result of the k-nearest neighbor classification method is: 84% accuracy in prediction of the target churn, 5.6% false negatives, 7.25% false positives, and a fitted model which can prove insightful to the research question.
Implications

The implications of a model which can accurately predict the target variable churn prove reliability within target exploration. The answers derived from the project are a better understanding of customer churn and which variables are most highly associated to churn. This allows for the company to make strategic management decisions to better mitigate churn. 

Limitation

Limitations of this classification method and model are found in the accuracy scores and data acquisition inaccuracies. There is sill a great deal of inaccuracy within the model, as it is wrong 16% of the time. Furthermore, the accuracy of the analysis in its entirety relies on the integrity of the data provided. The data being nonsensical throughout its various features proves an unstable ground for any model.  For example, the variable Area has three values: rural, suburban, and urban. Each of these three areas has almost identical descriptive statistics for population count. This is an example of nonsensical data. How does a city have the same population statistics as the countryside? They do not have similar populations; therefore, the data is nonsensical. Another example is bandwidth. This is a variable which should have been available to us through in-house repositories as it is a contractual service agreement that we provide different bandwidths based upon price. There are negative entries for this variable, which is again, nonsensical. The data acquisition phase should have caught this error and re-scraped the bandwidth from the repository. In summary, the limitations are grounded in model inaccuracy and inaccurate data.

Course of Action

The results of the project have provided answers to the research question and organizational goals of the analysis. The classification method and model are 84% accurate at predicting the target variable churn. Using this, the company should use strategic business management to mitigate churn by appealing to the customers which match the following predictors: tenure, bandwidth, longitude, latitude, zip code, monthly charge, streaming movies and tv, and contract. These are the customers that should be the focus of all mitigation and marketing efforts. The corresponding accuracy of the results of these mitigation efforts (based on the data provided) is 84%. Thus, the likelihood of accurately targeting customers who will churn with strategic management initiatives is 84%.


