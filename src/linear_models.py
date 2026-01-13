import numpy as np

class LinearRegressionOLS:
    """
    Production implementation of Ordinary Least Squares.
    Solves beta = (X'X)^-1 X'y
    """
    def __init__(self):
        self.coef_ = None
        
    def fit(self, X, y):
        # Add bias term (Intercept)
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Normal Equation
        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        
        try:
            self.coef_ = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback for singular matrix (Pseudo-inverse)
            self.coef_ = np.linalg.pinv(XtX) @ Xty
            
        return self
    
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.coef_

class RidgeRegression:
    """
    L2 Regularized Regression with Intercept Protection.
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        
    def fit(self, X, y):
        n_features = X.shape[1] + 1 # +1 for bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        
        I = np.eye(n_features)
        I[0, 0] = 0.0 # Do not penalize intercept
        
        XtX = X_b.T @ X_b
        Xty = X_b.T @ y
        
        self.coef_ = np.linalg.solve(XtX + self.alpha * I, Xty)
        return self
        
    def predict(self, X):
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b @ self.coef_
