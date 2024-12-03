import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# data dummy
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# model regresi linear
model = LinearRegression()
model.fit(X, y)

# Prediksi
print("Koefisien:", model.coef_)
print("Intercept:", model.intercept_)

# Plot data
plt.scatter(X, y, color='blue', label='Data Asli')
plt.plot(X, model.predict(X), color='red', label='Regresi Linear')
plt.legend()
plt.show()
