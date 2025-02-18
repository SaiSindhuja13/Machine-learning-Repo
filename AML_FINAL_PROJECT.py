#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


# In[19]:


divorce = pd.read_csv('/Users/sindhujaupadrashta/Downloads/archive (3)/Marriage_Divorce_DB.csv')
divorce


# In[20]:


# Display the first few rows of the dataset and a summary of its structure
info = divorce.info()


# In[21]:


data_head = divorce.head()
print(data_head)


# In[22]:


data_description = divorce.describe()
print(data_description)


# In[23]:


missing_values = divorce.isnull().sum()
print(missing_values)


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
#histograms for all features in dataset, giving us an insight into the distribution
divorce.hist(bins=15, figsize=(15, 10))
plt.show()
plt.figure(figsize=(10, 8))
#heatmap of Pearson correlation coefficients between all pairs of features
sns.heatmap(divorce.corr(), annot=True, cmap='coolwarm')
plt.show()


# In[25]:


divorce.drop_duplicates(inplace=True)


# In[26]:


# Assuming 'data' is your DataFrame containing the columns
X = divorce.drop(['Divorce Probability'], axis=1)  # Features
y = divorce['Divorce Probability']  # Target


# In[27]:


# Split the data into training and testing sets with 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[28]:


#Verify the shape of the datasets
(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# The data has been successfully split into training and testing sets, adhering to the 80% training and 20% testing split. The training set consists of 80 samples, and the testing set consists of 20 samples, with each sample having 30 features.

# # LINEAR REGRESSION MODEL

# It's a regression problem since we're predicting a continuous outcome (divorce probability).

# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Assuming X_train, X_test, y_train, y_test are already defined

# Initialize the Linear Regression model
linear_model = LinearRegression()

# Fit the model to the training data
linear_model.fit(X_train, y_train)

# Predict on the training and testing sets
y_train_pred = linear_model.predict(X_train)
y_test_pred = linear_model.predict(X_test)

# Calculate MSE and R-squared for both the training and testing sets
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("Linear Regression Performance:")
print("Training MSE:", train_mse)
print("Training R²:", train_r2)
print("Test MSE:", test_mse)
print("Test R²:", test_r2)


# Evaluate the model's performance on the test set using appropriate metrics, such as R-squared and Mean Squared Error (MSE).
# 
# MSE is a measure of the average squared difference between the observed actual outcomes and the outcomes predicted by the model, a lower MSE (closer to 0) is better.
# 
# R-squared is a statistical measure of how close the data are to the fitted regression line. A value which is close to 1 is better

# # RANDOM FOREST REGRESSION MODEL

# In[30]:


# Initialize the Random Forest Regressor
random_forest_regressor = RandomForestRegressor(n_estimators=100, random_state=42)


# In[31]:


# Train the model on the training data
random_forest_regressor.fit(X_train, y_train)


# In[32]:


# Predict on the training and testing sets
y_train_pred = random_forest_regressor.predict(X_train)
y_test_pred = random_forest_regressor.predict(X_test)


# # Evaluate the model

# In[33]:


# Evaluate the model
train_mse = mean_squared_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)


# In[34]:


(train_mse, train_r2, test_mse, test_r2)
print("Random forest Regressor Performance:")
print(f"train MSE: {train_mse}")
print(f"train r square: {train_r2}")
print(f"test MSE: {test_mse}")
print(f"test r square: {test_r2}")


# The Random Forest Regressor model has been trained, and we've evaluated its performance on both the training and testing sets.
# 

# On the training set, the model shows good performance with an R² score of 0.8, indicating that it can explain about 84.2% of the variance in divorce probability. However, the model's performance drops significantly on the test set, with an R² score close to zero, which suggests it doesn't generalize well to unseen data.

# This discrepancy between training and testing performance could indicate overfitting, where the model learns the training data too closely and fails to make accurate predictions on new, unseen data. A negative R² score on the testing set also implies that the model is performing worse than a simple mean-based prediction.

# # Improve the model

# To improve the model, we might consider several approaches:
# 
# we are using hyper parameter tuning and cross validation

# 4.1 Hyper parameter tuning

# The goal is to find a set of hyperparameters that improves the model's ability to generalize to unseen data. We'll use a grid search approach with cross-validation to search through a predefined space of hyperparameters and find the best combination based on a scoring metric, such as R² or negative mean squared error.

# In[35]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor

# Initialize the model
model = GradientBoostingRegressor(random_state=42)

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 4],
    'min_samples_leaf': [1, 2],
    'subsample': [0.9, 1.0]  # Fraction of samples to be used for fitting the individual base learners
}

# Set up the grid search with cross-validation
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)

# Perform the grid search
grid_search.fit(X, y)

# Best parameters found
print("Best parameters:", grid_search.best_params_)

# Best score found
print("Best score (negative MSE):", grid_search.best_score_)


# this MSE score is negative because GridSearchCV uses negative values for loss functions by convention, where a higher (less negative) value indicates a better model.
# 

# With these optimized parameters, our model is better configured to learn from the training data without fitting too noisily or too smoothly.
# The learning rate indicates how much the model is adjusted at each step of the boosting process, a slower rate (0.01) suggesting that more trees (n_estimators) are needed for the model to learn effectively.
# However, 100 trees were found to be optimal within the search space. 
# The subsample rate of 0.9 suggests that using a fraction of the data (90%) for fitting each base learner helps prevent overfitting.
# 
# To proceed, we can retrain the Gradient Boosting Regressor using these parameters on the full training dataset, then evaluate its performance on test set or through cross-validation to confirm its effectiveness.

# In[36]:


# Retrain with the best parameters
optimized_model = GradientBoostingRegressor(
    learning_rate=0.01,
    max_depth=3,
    min_samples_leaf=1,
    min_samples_split=2,
    n_estimators=100,
    subsample=0.9,
    random_state=42
)

optimized_model.fit(X_train, y_train)  



# In[37]:


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


y_pred = optimized_model.predict(X_test)

# Calculate the evaluation metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Gradient boosting Regressor performance:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")


# By optimizing hyperparameters, the model likely became better at generalizing, as indicated by the improved and now positive test R², though it remains relatively low.
# While the improvements are a step in the right direction, the overall model performance, particularly the R² value, suggests there's still room for improvement in how well the model can predict unseen data

# 4.2 Cross validation 

# In[38]:


from sklearn.model_selection import cross_val_score
import numpy as np

# Configure the number of folds
n_folds = 5

# Perform cross-validation for R-squared (R²)
cv_r2_scores = cross_val_score(optimized_model, X, y, cv=n_folds, scoring='r2')

# Perform cross-validation for MSE
cv_mse_scores = -cross_val_score(optimized_model, X, y, cv=n_folds, scoring='neg_mean_squared_error')

print(f"Average R-squared (R²) across {n_folds} folds: {np.mean(cv_r2_scores)}")
print(f"Average Mean Squared Error (MSE) across {n_folds} folds: {np.mean(cv_mse_scores)}")


# Didnt work!

# given that the training performance is not very high, it's more indicative of the model not capturing the underlying relationship between the features and the target variable well.. also, The drastic drop in performance from training to testing is usually a sign of overfitting;
XGBoost Performance:
Training MSE: 0.15745791289361386
Training R²: 0.49336960475389247
Test MSE: 0.4123071973094284
Test R²: -0.2664881977407121
# # Using lightgbm algorithm

# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

divorce = pd.read_csv('/Users/sindhujaupadrashta/Downloads/archive (3)/Marriage_Divorce_DB.csv')

X = divorce.drop(['Divorce Probability'], axis=1)  # Features
y = divorce['Divorce Probability']  # Target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train.select_dtypes(include=['object', 'bool']).columns


numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)])


model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LGBMRegressor(random_state=42))])


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


r2 = r2_score(y_test, y_pred)
print(f'R^2 Score: {r2}')


# In[ ]:




