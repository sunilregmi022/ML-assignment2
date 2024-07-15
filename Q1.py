import numpy as np

def compute_mean(data):
    """Compute the mean of a list of numbers."""
    return np.mean(data)

def compute_variance(data):
    """Compute the variance of a list of numbers."""
    return np.var(data, ddof=1)

def compute_covariance(data1, data2):
    """Compute the covariance between two lists of numbers."""
    return np.cov(data1, data2, ddof=1)[0, 1]

def gaussian_distribution(x, mean, variance):
    """Compute the Gaussian distribution value for a given x, mean, and variance."""
    return (1.0 / np.sqrt(2 * np.pi * variance)) * np.exp(-((x - mean) ** 2) / (2 * variance))


data = [25, 50, 60, 80, 66]
data1 = [32, 42, 53, 64, 65]
data2 = [20, 30, 40, 50, 60]

mean = compute_mean(data)
variance = compute_variance(data)
covariance = compute_covariance(data1, data2)
gaussian_value = gaussian_distribution(3, mean, variance)

print(f"Mean: {mean}")
print(f"Variance: {variance}")
print(f"Covariance: {covariance}")
print(f"Gaussian value at x=3: {gaussian_value}")
