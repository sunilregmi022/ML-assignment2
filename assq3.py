import numpy as np
from cvxopt import matrix, solvers
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self, C=1.0, kernel='linear', degree=3, gamma='scale'):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.alpha = None
        self.b = 0
        self.X = None
        self.y = None

    def _kernel(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2.T)
        elif self.kernel == 'poly':
            return (1 + np.dot(x1, x2.T)) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma = 1 / (x1.shape[1] * np.var(self.X))
            else:
                gamma = self.gamma
            return np.exp(-gamma * np.linalg.norm(x1[:, None] - x2, axis=2) ** 2)

    def fit(self, X, y):
        self.X = X
        self.y = y
        n, m = X.shape

        K = self._kernel(X, X)

        P = matrix(np.outer(y, y) * K)
        q = matrix(-np.ones(n))
        G_std = np.diag(np.ones(n) * -1)
        h_std = np.zeros(n)
        G_slack = np.diag(np.ones(n))
        h_slack = np.ones(n) * self.C
        G = matrix(np.vstack((G_std, G_slack)))
        h = matrix(np.hstack((h_std, h_slack)))
        A = matrix(y.astype(np.double), (1, n), 'd')
        b = matrix(0.0)

        solution = solvers.qp(P, q, G, h, A, b)
        self.alpha = np.ravel(solution['x'])

        sv = self.alpha > 1e-5
        ind = np.arange(len(self.alpha))[sv]
        self.alpha = self.alpha[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]

        self.b = 0
        for n in range(len(self.alpha)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.alpha * self.sv_y * K[ind[n], sv])
        self.b /= len(self.alpha)

        if self.kernel == 'linear':
            self.w = np.zeros(m)
            for n in range(len(self.alpha)):
                self.w += self.alpha[n] * self.sv_y[n] * self.sv[n]

    def predict(self, X):
        if self.kernel == 'linear':
            return np.sign(np.dot(X, self.w) + self.b)
        else:
            y_predict = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                s = 0
                for alpha, sv_y, sv in zip(self.alpha, self.sv_y, self.sv):
                    s += alpha * sv_y * self._kernel(X[i:i+1], sv)
                y_predict[i] = s
            return np.sign(y_predict + self.b)

# Generate synthetic data
X, y = make_blobs(n_samples=100, centers=2, random_state=6)
y = np.where(y == 0, -1, 1)  # Convert labels to -1, 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train SVM
svm = SVM(C=1.0, kernel='linear')
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)

print("Predictions:", y_pred)
print("True labels:", y_test)

