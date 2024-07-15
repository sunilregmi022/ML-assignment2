import numpy as np

def calculate_entropy(y):
    """
    Calculate the entropy of the target labels.

    Parameters:
    y : numpy array of shape [number of samples]
        The target labels.

    Returns:
    float : The entropy of the labels.
    """
    class_counts = np.bincount(y)
    probabilities = class_counts / len(y)
    entropy = -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    return entropy

def calculate_information_gain(X, y, columns: list, attribute: str) -> float:
    """
    Calculate the information gain of a given attribute in the dataset.

    Parameters:
    X : numpy array of shape [number of samples x number of features]
        The feature matrix.
    y : numpy array of shape [number of samples]
        The target labels.
    columns: list of str
        The list of column names corresponding to the features.
    attribute: str
        The column name for which we need to calculate the information gain.

    Returns:
    float : The information gain of the given attribute.
    """
    # Find the index of the given attribute
    attribute_index = columns.index(attribute)
    
    # Calculate the entropy of the whole dataset
    total_entropy = calculate_entropy(y)
    
    # Get the unique values of the attribute
    unique_values = np.unique(X[:, attribute_index])
    
    # Calculate the weighted entropy of the subsets
    weighted_entropy = 0
    for value in unique_values:
        subset_mask = X[:, attribute_index] == value
        subset_y = y[subset_mask]
        subset_entropy = calculate_entropy(subset_y)
        weighted_entropy += (len(subset_y) / len(y)) * subset_entropy
    
    # Calculate the information gain
    gain = total_entropy - weighted_entropy
    
    return gain

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])
columns = ['feature1', 'feature2']
attribute = 'feature1'
print(f"Information Gain for {attribute}: {calculate_information_gain(X, y, columns, attribute)}")
