import numpy as np

def sigmoid_function(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression():

    def __init__(self, learning_rate = 1e-3, iterations = 1000) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None
    
    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            linear_preds = np.dot(X, self.weights) + self.bias
            predictions = sigmoid_function(linear_preds)

            dw = (1/n_samples) * np.dot(X.T, (predictions - Y))
            db = (1/n_samples) * np.sum(predictions - Y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        linear_preds = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid_function(linear_preds)
        class_predicted = [0 if y <= 0.5 else 1 for y in y_predicted]
        return class_predicted