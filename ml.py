##machine learning code 


# Linear Regression 
# Dataset: cs448b_ipasn.csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# 1) Load the data

df= pd.read_csv("cs448b_ipasn.csv")
df.columns = df.columns.str.strip()

# define input x and output y

X = df[["l_ipn"]]
y = df["f"]

#creat and train model
model=LinearRegression()
model.fit(X,y)

# Smooth line
x_line = np.linspace(X["l_ipn"].min(), X["l_ipn"].max(), 100).reshape(-1, 1)
y_line = model.predict(x_line)

#results
plt.figure()
plt.scatter(X["l_ipn"] , y)
plt.plot(x_line, y_line)
plt.xlabel("local IP Number")
plt.ylabel("number of flows")
plt.title("linear regression: IP vs Network Flowa")
plt.show()

#print regression equation
print("LINEAR REGRESSION EQUATION:")
print(f"Flows = {model.coef_[0]:.2f} * l_ipn + {model.intercept_:.2f}")
