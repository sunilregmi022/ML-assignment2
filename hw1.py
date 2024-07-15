import numpy as np

def calculate_entropy(X, y):
    """
    Calculate the entropy of the given dataset.

    Parameters:
    X : numpy array of shape [number of samples x number of features]
        The feature matrix.
    y : numpy array of shape [number of samples]
        The target labels.

    Returns:
    float : The entropy of the dataset.
    """
    # Calculate the frequency of each class in the target variable
    class_counts = np.bincount(y)
    
    # Calculate the probabilities of each class
    probabilities = class_counts / len(y)
    
    # Calculate the entropy
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    return entropy

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])
print(calculate_entropy(X, y))
