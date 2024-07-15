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

def calculate_information_gain(X, y, columns, attribute):
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
    attr_index = columns.index(attribute)
    total_entropy = calculate_entropy(y)
    values, counts = np.unique(X[:, attr_index], return_counts=True)
    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset_indices = np.where(X[:, attr_index] == value)
        subset_y = y[subset_indices]
        subset_entropy = calculate_entropy(subset_y)
        weighted_entropy += (count / len(y)) * subset_entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

def id3(X, y, columns, depth=0):
    """
    Implement the ID3 algorithm to create a decision tree.

    Parameters:
    X : numpy array of shape [number of samples x number of features]
        The feature matrix.
    y : numpy array of shape [number of samples]
        The target labels.
    columns: list of str
        The list of column names corresponding to the features.
    depth: int
        The current depth of the tree. Used for printing the tree structure.

    Returns:
    dict or int : The decision tree or the target value at the leaf node.
    """
    # If all target values are the same, return a leaf node with that value
    unique_labels = np.unique(y)
    if len(unique_labels) == 1:
        return unique_labels[0]

    # If there are no more columns to split on, return the most common target value
    if len(columns) == 0:
        return np.bincount(y).argmax()

    # Find the attribute with the highest information gain
    gains = [calculate_information_gain(X, y, columns, col) for col in columns]
    best_attr_index = np.argmax(gains)
    best_attr = columns[best_attr_index]

    # Create the root node with the best attribute
    tree = {best_attr: {}}
    remaining_columns = columns[:best_attr_index] + columns[best_attr_index + 1:]

    # Split the dataset based on the best attribute and recursively create branches
    attr_index = columns.index(best_attr)
    values = np.unique(X[:, attr_index])
    for value in values:
        subset_indices = np.where(X[:, attr_index] == value)
        subset_X = X[subset_indices]
        subset_y = y[subset_indices]
        subtree = id3(subset_X, subset_y, remaining_columns, depth + 1)
        tree[best_attr][value] = subtree

    return tree

# Example usage
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([0, 1, 0, 1])
columns = ['Feature1', 'Feature2']
tree = id3(X, y, columns)
print(tree)
