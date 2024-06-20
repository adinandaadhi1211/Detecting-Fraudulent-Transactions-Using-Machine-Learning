# EMPOWERING FINANCIAL SECURITY : DETECTING FRAUDULENT TRANSACTIONS 

## INTRODUCTION
	

### Project Overview

	This project aims to build a best model for detecting fraudulent transactions.
   Five algorithms were implemented: Logistic Regression, Decision Tree, Random Forest,
	SVM(Support Vector Machine) and KNN(K-Nearest Neighbours).
   Hyperparameter Tuning and Oversampling methods were applied to enhance model performance.

### Importance of Detecting Fraudulent Transactions

	Fraud detection is crucial to protect businesses and individuals from financial losses.
   Efficiently identifying fraudulent transactions help maintain trust in financial systems.
   Early detection of fraud can prevent significant damage and legal complications.

### IMBALANCED DATASET
	
	Fraudulent transactions are typically much rarer than legitimate transactions, leading to an imbalanced 
Dataset. This imbalance can cause machine learning models to be biased towards predicting the majority
Class (legitimate transactions) resulting in poor performance in detecting fraudulent transactions.


### COMPLEX PATTERNS

	Fraudulent activities often involve complex patterns that can be difficult to detect using simple rules.
Advanced machine learning techniques are needed to capture these patterns effectively. 


### REAL-TIME DETECTION

	Ideally, the model should be able to detect fraudulent transactions in real-time to prevent fraudulent
Activities before they result in significant losses.

## DATA DESCRIPTION


 ### Dataset Information

 Original Size of the dataset : 39221 rows and 8 columns.

 Target column distribution : Legitimate transaction(0) – 38661 and Fraudulent Transactions(1) – 560 (Highly Imbalanced).

 Mostly have categorical variables and 3 continuous variables.

 Category has 0.24% and IsWeekend has 1.43% of missing values.

 In this dataset we have 3033 duplicate values.

## EXPLORATORY DATA ANALYSIS

### DATA VISUALIZATION
![image](https://github.com/adinandaadhi1211/Detecting-Fraudulent-Transactions-Using-Machine-Learning/assets/128919839/9e207898-4db2-406f-ad5a-03799d3b3169)

![image](https://github.com/adinandaadhi1211/Detecting-Fraudulent-Transactions-Using-Machine-Learning/assets/128919839/da4cd4e5-a4ba-4871-bdbb-d75d965c5c29)

![image](https://github.com/adinandaadhi1211/Detecting-Fraudulent-Transactions-Using-Machine-Learning/assets/128919839/6682ce16-0dfa-4436-8758-b3176561dd9d)

![image](https://github.com/adinandaadhi1211/Detecting-Fraudulent-Transactions-Using-Machine-Learning/assets/128919839/0985618a-49b5-432e-bc77-8d0ad0fa02cd)

![image](https://github.com/adinandaadhi1211/Detecting-Fraudulent-Transactions-Using-Machine-Learning/assets/128919839/bdc9171d-53bc-499f-a867-4a2c6b38f16e)

![image](https://github.com/adinandaadhi1211/Detecting-Fraudulent-Transactions-Using-Machine-Learning/assets/128919839/d9b83387-26fb-4b12-a4a3-b565226f50ad)

## DATA PREPROCESSING
### Handling Missing Values

  Category : 0.24% missing values
   IsWeekend : 1.43% missing values
   Method : Handled using Mode

### Removing Duplicates

  Total duplicates removed  : 3033

### Handling Outliers
	
	 Columns with outliers : numItems, localTime and paymentMethodAgeDays.
    Method : Replaced with fenced IQR values.

### Encoding Categorical Variables

### Encoded the categorical variables : PaymentMethod, Category.
 Method : One-hot Encoding

 Missing values : Imputed using Mode.

 Duplicates : Removed 3033 duplicate entries.

 Outliers : Handled using the Interquartile Range (IQR) Method.

 Encoding : Applied One-hot encoding to Categorical variables

 Feature Engineering : Created new columns hour and minutes with existing column localTime.

 Balancing Data : Used SMOTETomek to handle class imbalance.

## SMOTETomek OVERSAMPLING TECHNIQUE

	SMOTETomek is a hybrid approach that combines SMOTE(Synthetic Minority Over-sampling 
Technique) and Tomek links. This technique is used to handle imbalanced datasets by both increasing the 
number of minority class samples and cleaning the dataset to improve the performance of Machine 
Learning models.
 To clean the dataset by removing ambiguous samples that are close to the decision boundary between
      classes.
Cleans the dataset by removing borderline cases.
Improves the clarity of the decision boundary between classes.

## APPLICATION IN THIS PROJECT
	
	SMOTETomek was applied to balance the fraudulent transaction dataset. This helped in mitigating the class imbalance issue, leading to improved performance metrics for the models.
Shape of the data before sampling : (36175, 9)
Shape of the data after sampling : (71568, 9)

## ALGORITHMS IMPLEMENTED

  LOGISTIC REGRESSION

  DECISION TREE

  RANDOM FOREST

  SUPPORT VECTOR MACHINE (SVM)

  K-NEAREST NEIGHBOURS (KNN)

## LOGISTIC REGRESSION

### Metrics :
	
	 Accuracy : 0.96

   Precision : (0) – 1.00 and (1) – 0.93

   Recall : (0) – 0.92 and (1) – 1.00

   f1-score : (0) – 0.96 and (1) – 0.96

### Confusion matrix : 

	[[9898       830],
	  [ 0       10743]]

## LOGISTIC REGRESSION – AFTER TUNING

### Metrics : 

	 Accuracy : 0.99

   Precision : (0) – 1.00 and (1) – 0.97

   Recall : (0) – 0.97 and (1) – 1.00

   f1-score : (0) – 0.99 and (1) – 0.99

   ROC AUC Score : 0.99

### Confusion Matrix : 

	[[10419     309],
	  [0        10743]]

## DECISION TREE

### Metrics : 

 	 Accuracy : 1.00

   Precision : (0) – 1.00 and (1) – 1.00

   Recall : (0) – 1.00 and (1) – 1.00

   f1-score : (0) – 1.00 and (1) – 1.00

## Confusion matrix : 

	[[10728         0],
	  [ 0       10743]]

## DECISION TREE AFTER TUNING

### Metrics : 

 	 Accuracy : 1.00

   Precision : (0) – 1.00 and (1) – 1.00

   Recall : (0) – 1.00 and (1) – 1.00

   f1-score : (0) – 1.00 and (1) – 1.00

   ROC AUC Score : 1.00

### Confusion matrix : 

	[[10728         0],
	  [ 0       10743]]

## RANDOM FOREST

### Metrics : 

	 Accuracy : 1.00

   Precision : (0) – 1.00 and (1) – 1.00

   Recall : (0) – 1.00 and (1) – 1.00

   f1-score : (0) – 1.00 and (1) – 1.00

### Confusion matrix : 

	[[10728         0],
	  [ 0       10743]]

## RANDOM FOREST AFTER TUNING

### Metrics : 

 	 Accuracy : 1.00

   Precision : (0) – 1.00 and (1) – 1.00

   Recall : (0) – 1.00 and (1) – 1.00

   f1-score : (0) – 1.00 and (1) – 1.00

   ROC AUC Score : 1.00

### Confusion matrix : 

	[[10728         0],
	  [ 0       10743]]

## SVM

### Metrics : 

 	 Accuracy : 0.96

   Precision : (0) – 1.00 and (1) – 0.92

   Recall : (0) – 0.91 and (1) – 1.00

   f1-score : (0) – 0.95 and (1) – 0.96

### Confusion matrix : 

	[[9786       942],
	  [ 12      10731]]

## KNN

### Metrics : 

 	 Accuracy : 1.00

   Precision : (0) – 1.00 and (1) – 0.99

   Recall : (0) – 0.99 and (1) – 1.00

   f1-score : (0) – 1.00 and (1) – 1.00

### Confusion matrix : 

	[[10644         84],
	  [ 2         10741]]

## KNN AFTER TUNING

### Metrics : 

 	 Accuracy : 1.00

   Precision : (0) – 1.00 and (1) – 1.00

   Recall : (0) – 1.00 and (1) – 1.00

   f1-score : (0) – 1.00 and (1) – 1.00

   ROC AUC Score : 1.00

### Confusion matrix : 

	[[10705        23],
	  [ 2         10741]]

## HYPERPARAMETER TUNING

	Hyperparameter tuning, also known as Hyperparameter Optimization, is the process of finding the best set of hyperparameters for a machine learning model. Hyperparameters are configuration settings of a model that are set before the training process begins and remain fixed during training. 
Hyperparameter tuning aims to search through different combinations of hyperparameter values to find the optimal configuration that maximizes the model’s performance on the validation set. The goal is to strike a balance between underfitting and overfitting.

## GridSearchCV

	Grid search involves defining a grid of possible hyperparameter values and evaluating the model’s performance for all possible combinations within the grid. It can be computationally expensive but exhaustive in searching the hyperparameter space.

## RESULT
 Logistic Regression shows a significant improvement after tuning, reaching near-perfect scores in accuracy,  Precision, recall, and f1-score. Its ROC AUC score of 0.99 suggests excellent discrimination 
 capability. 

 Decision Tree and Random Forest method achieve perfect scores across all metrics after tuning, indicating very strong performance. Their ROC AUC scores are also close to 1.00, suggesting excellent 
 discrimination capabilities. However, these perfect scores raise concerns about overfitting.

 SVM performs well with an accuracy of 0.96, though its precision and recall scores are slightly lower compared to the Decision tree and Random forest models. Its ROC AUC score is 0.96, indicating good 
 discrimination capability but not as strong as other models.

 KNN also achieves perfect scores across all metrics after tuning, similar to the Decision Tree and Random Forest models. Its ROC AUC score is 1.00, suggesting excellent discrimination capability. Howerver like 
 others , its perfect score raise concerns about overfitting.

## Identifying Overfitting Models

Overfitting typically occurs when a model learns the training data too well, capturing noise along with the underlying pattern, leading to poor generalization to new, unseen data. Given the perfect scores achieved by the Decision tree, Random Forest and KNN models, there is a significant risk of overfitting.

Decision Tree, despite being simple and interpretable, can easily overfit due to its tendency to create complex , high variance trees. Its perfect scores suggest it might be fitting the noise in the training data.

Random Forest reduces overfitting by averaging the predictions of multiple trees, but its perfect scores still hint at potential overfitting , especially if the model complexity doesn’t match the complexity of the underlying data.

KNN, particularly with a very high value of k (considered after tuning), might capture noise in the training data, leading to perfect scores that don’t generalize well to unseen data.

## BEST MODEL SELECTION

After the analysis, the Logistic Regression (after tuning) model seems to be the best choice among the five models, despite its slightly lower performance compared to the Decision tree, Random forest and KNN models. It strikes a better balance between performance and the risk of overfitting , with its near – perfect scores and a slightly lower ROC AUC score indicating a more cautious approach to learning from the data.










