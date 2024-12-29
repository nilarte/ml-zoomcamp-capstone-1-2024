# ml-zoomcamp-capstone-1-2024
## Description
This is DataTalksClub [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) capstone 1 project repo.

It uses simple employee dataset from IT industry (presumably from India). 
This model tries to predict if employee would leave the company in next 2 years. 

Dataset credit: https://www.kaggle.com/datasets/tejashvi14/employee-future-prediction

Here are step by step details about how we build optimal model to predict employee leaving probability. 

## 1. Data preparation, cleanup and EDA
Code here: [notebook.ipynb](./notebook.ipynb)

Parse downloaded dataset [Employee.csv](./Employee.csv) via `pandas`.

Note: We are using local dataset copy here but we can download data from kaggle in notebook as well.
```python
kagglehub.dataset_download("tejashvi14/employee-future-prediction")
```
Look for NAN values in data (There are none). 

Feature importance of rest features with our target variable: leaveornot:

We find mutual info for categorical features and correlation for numerical features.

Top 3 most relevant categorical fetatures are
```bash
gender         0.024192
city           0.021632
education      0.010592
```

Top 3 most relevant numerical fetatures are
```bash
paymenttier                  0.197638
joiningyear                  0.181705
age                          0.051126
```

## 2. Training a model
Code here: [notebook.ipynb](./notebook.ipynb)
### 2.1 One-hot encoding
Turn categorical data into binary vector
### 2.2 Simple Logistic regression
Train a simple logistic regression model.

### 2.2 Cross-Validation with kfold
Train same model with different values of kfold.

Check AUC score for validation data.
### 2.3 Random forest regressor
Train random regression model.
Try different values of max_depth, n_estimators and min_samples_leaf.
Select values giving the best AUC score via graph.

![rf-graph.png](./rf-graph.png)

### 2.4 XGBoost
Train random regression model.
Try different values of max_depth, eta and min_child_weight.
Select values giving the best AUC score.

### 2.5 Selecting the best model
Random forest model is giving slightly better results.
We also checked with test data.