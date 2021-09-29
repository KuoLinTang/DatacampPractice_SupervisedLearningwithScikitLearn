from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import dataset
boston = datasets.load_boston()

# Exploring dataset
print(boston["data"].shape)
print(boston.keys())

# Rearrange dataset
df1 = pd.DataFrame(boston["data"], columns = boston['feature_names'])
df2 = pd.DataFrame(boston["target"], columns = np.array(["MEDV"]))
df = pd.concat([df1.reset_index(drop=True), df2], axis=1)
df.index = range(506)
print(df.head(5))


# Split dataset into independent variable set and dependent variable (Label) set
X = df.drop("MEDV", axis = 1).to_numpy()
y = df["MEDV"].to_numpy()
# to_numpy代表將pandas的dataframe轉換成numpy array
# 如果前面是Series，也可以用.values()


# =================================================================
print("=================================================================")
# 首先，用RM這個column來predict target variable MEDV
## 取出RM這個column
X_RM = X[:, 5]
print(X_RM)
## 此時X_RM是numpy的1D array，必須轉換成2D array才能丟進regression裡面
X_RM_2 = X_RM.reshape(-1,1)
y_2 = y.reshape(-1,1)

## Plot a scatter plot y_2 by X_RM_2
plt.scatter(X_RM_2, y_2)
plt.xlabel("Number of rooms")
plt.ylabel("Value of house /1000 ($)")
plt.show()

## 上圖看起來X_RM_2跟y_2有正相關，因此來回歸分析一下
from sklearn.linear_model import LinearRegression

## 訓練模型
reg = LinearRegression()
reg.fit(X_RM_2, y_2)

## 預測數值
## 建立用來預測的資料
prediction_data = np.linspace(min(X_RM_2), max(X_RM_2)).reshape(-1,1)

## 預測
reg.predict(prediction_data)

# R square
print(reg.score(X_RM_2, y_2))


# Remake the plot
plt.figure()
# Add scatter
plt.scatter(X_RM_2,y_2,color = "blue")                     
# Add Regression line
plt.plot(prediction_data, reg.predict(prediction_data), color = "black", linewidth = 3)
plt.show()



# =================================================================
print("=================================================================")

# Cross-Validation

## 先拆Training set和Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=21, stratify = y)


# Refresh regression object
reg = LinearRegression()

# Cross-validation
from sklearn.model_selection import cross_val_score
cv_results = cross_val_score(reg, X_train, y_train, cv = 5)

print(cv_results)
print("Average 5-Fold CV Score: {}".format(np.mean(cv_results)))