# Fault_Detection

## Project Overview
This project is developed for the **Sensor Fault Detection challenge**.  
The objective is to predict whether a device is operating normally or showing a faulty condition using the sensor measurements provided in the dataset.
Working with sensor data is not always straightforward. Two main challenges were observed while working on the dataset:
• Sensor readings can contain noise or unexpected spikes.  
• Fault cases are much fewer compared to normal operating conditions.
Because of these challenges,I designed a structured machine learning pipeline that focuses on proper preprocessing,feature engineering,model selection,and tuning.
The models are evaluated using **Macro F1-Score**, which is the official evaluation metric used in the challenge.

---

## Dataset
The dataset provided for the challenge contains two files:
• **TRAIN.csv** – Used to train the machine learning models.  
• **TEST.csv** – Used to generate predictions for evaluation.
Each record contains 47 numerical features (F01 – F47). These features represent quantitative measurements captured by the monitoring system during device operation.

The target variable is Class, which indicates the operational status of the device:
0 → Device operating under normal condition 
1 → Device exhibiting a faulty condition

---

# Model Architecture Pipeline

# Input Data (TRAIN.csv)

Features: F01 – F47 
Target Variable: Class
0 → Device operating under normal conditions  
1 → Device exhibiting a faulty condition

---

# Data Preprocessing

Before training the models, the data was cleaned and prepared.

• Outlier handling: Extreme values were limited using percentile capping.  
• Feature scaling:RobustScaler was used to reduce the effect of extreme values.

---
# Feature Engineering

To help the models understand the sensor behaviour better,additional features are created.
These include:
• Mean of the sensor readings  
• Standard deviation of sensor values  
• Differences betwen consecutive sensor measurements  
• Deviation of each sensor value from the overall sensor mean

These features help capture patterns in the sensor data that may indicate faulty device behaviour.

---

# Feature Selection

After feature engineering, the number of features increased significantly.

To reduce unnecessary features, a LightGBM model was trained to compute feature importance scores.  
The least important features were removed so that the final models focus only on the most useful signals.

---

# Model Selection

During experimentation, multiple machine learning algorithms were tested using cross-validation to see which models work best on the dataset.
The models evaluated include:

• AdaBoost  
• Gradient Boosting  
• Random Forest  
• Decision Tree  
• K-Nearest Neighbors  
• Logistic Regression  
• Naive Bayes  
• Support Vector Machine  
• XGBoost  
• CatBoost  
• LightGBM  
• HistGradientBoosting  

All models were evaluated using **Macro F1-Score**.
Based on the results, the strongest performing models were selected for further tuning.

---
# Model Tuning (Hyperparameter Optimization)

The selected models were tuned using Optuna.
Optuna automatically searches for better hyperparameter combinations by running multiple trials and selecting the ones that improve the evaluation score.
The following models were optimized:

• XGBoost  
• HistGradientBoosting  
• CatBoost  
• Random Forest  
The tuning procss focused on maximizing **Macro F1-Score**.

---
# Stacking Ensemble Model
Instead of relying on single model,the tuned models are combined using stacking ensemble.
Base Models:
• XGBoost  
• HistGradientBoosting  
• CatBoost  
• Random Forest  

Final Meta Model:
• Logistic Regression
The Logistic Regression model takes predictions from base models and learns how to combine them to produce final prediction.

---

# Cross-Validation Training
To properly evaluate model performance, **5-Fold Stratified Cross-Validation** is used.
This ensures that both classes remain balanced in every fold:
• Class = 0 → Normal devices  
• Class = 1 → Faulty devices
Scaling and training are performed inside each fold to avoid data leakage

---

# Threshold Optimization
The default probability threshold for classification is usually 0.5, but this may not be optimal when the dataset is imbalanced.
Different thresholds between 0.1 and 0.9 were tested to find the value that gives the best **Macro F1-Score**

---

# Model Explainability
To better understand the model's behavior, SHAP (SHapley Additive Explanations)used.
SHAP helps visualize how different features(F01 – F47) contribute to the model’s predictions.

---

# Final Prediction Generation
The trained stacking model is used to generate predictions for **TEST.csv**.
The predicted probabilities were converted into **Class labels (0 or 1)** using the optimized threshold.

---

# Submission File
Predictions are saved in **FINAL.csv** using required submission format:
ID → CLASS

---

## Implementation Notes
During experimentation, I tested several models using cross-validation to understand how they perform on dataset.
In my experiments, **tree-based models** consistently performed better than other algorithms. They were able to capture complex relationships between sensor features more effectively.
Combining multiple models using a **stacking approach** also improved stability of predictions and slightly increased overall **Macro F1-Score** compared to using a single model.

---

## Tools & Libraries Used

Python  
Pandas / NumPy  
Scikit-learn  
XGBoost  
CatBoost  
LightGBM  
Optuna  
SHAP  

---
## Result

The final stacking model achieved a **Macro F1-Score of approximately 98% during cross-validation**, indicating strong performance on the dataset.
