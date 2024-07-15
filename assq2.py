import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# New Dataset
years = np.array([2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019])
snowfall = np.array([50.1, 48.6, 47.3, 45.9, 44.5, 43.2, 41.8, 40.5, 39.1, 37.8, 36.4, 35.1, 33.7, 32.4, 31.0])

# Normalize the data
years_mean = np.mean(years)
years_std = np.std(years)
snowfall_mean = np.mean(snowfall)
snowfall_std = np.std(snowfall)

X = (years - years_mean) / years_std
Y = (snowfall - snowfall_mean) / snowfall_std

# Initialize parameters
theta0 = 0  # intercept
theta1 = 0  # slope
alpha = 0.01  # learning rate
iterations = 1000  # number of iterations

# Hypothesis function
def hypothesis(x, theta0, theta1):
    return theta0 + theta1 * x

# Cost function
def compute_cost(X, Y, theta0, theta1):
    m = len(Y)
    total_cost = (1 / (2 * m)) * np.sum((hypothesis(X, theta0, theta1) - Y) ** 2)
    return total_cost

# Gradient descent algorithm
def gradient_descent(X, Y, theta0, theta1, alpha, iterations):
    m = len(Y)
    cost_history = np.zeros(iterations)

    for i in range(iterations):
        theta0 = theta0 - alpha * (1 / m) * np.sum(hypothesis(X, theta0, theta1) - Y)
        theta1 = theta1 - alpha * (1 / m) * np.sum((hypothesis(X, theta0, theta1) - Y) * X)
        cost_history[i] = compute_cost(X, Y, theta0, theta1)

    return theta0, theta1, cost_history

# Training the model
theta0, theta1, cost_history = gradient_descent(X, Y, theta0, theta1, alpha, iterations)

# Print the final parameters
print(f"Theta0 (intercept): {theta0}")
print(f"Theta1 (slope): {theta1}")

# Predict the snowfall for 2020 using our model
year_2020_normalized = (2020 - years_mean) / years_std
snowfall_2020_normalized = hypothesis(year_2020_normalized, theta0, theta1)
snowfall_2020 = snowfall_2020_normalized * snowfall_std + snowfall_mean
print(f"Predicted snowfall for 2020 (custom model): {snowfall_2020} inches")

# Plotting the cost function history
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function History')
plt.show()

# Plotting the regression line
plt.scatter(X, Y, color='blue', label='Data Points')
plt.plot(X, hypothesis(X, theta0, theta1), color='red', label='Regression Line')
plt.xlabel('Years (normalized)')
plt.ylabel('Snowfall (inches, normalized)')
plt.title('Snowfall Data and Linear Regression Line')
plt.legend()
plt.show()

# Verifying with scikit-learn
X_reshaped = years.reshape(-1, 1)  # Reshape for sklearn
model = LinearRegression()
model.fit(X_reshaped, snowfall)

# Predicting for 2020 using scikit-learn model
snowfall_2020_sklearn = model.predict([[2020]])[0]
print(f"Predicted snowfall for 2020 (scikit-learn model): {snowfall_2020_sklearn} inches")

# Plotting the scikit-learn regression line
plt.scatter(years, snowfall, color='blue', label='Data Points')
plt.plot(years, model.predict(X_reshaped), color='green', label='sklearn Regression Line')
plt.xlabel('Years')
plt.ylabel('Snowfall (inches)')
plt.title('Snowfall Data and Linear Regression Line (scikit-learn)')
plt.legend()
plt.show()
