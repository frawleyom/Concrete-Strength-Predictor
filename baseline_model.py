from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import time

def run_linear_regression(X_train, X_test, y_train, y_test):

    # Start timing
    start_time = time.time()    

    # Initialize and train
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    # Predict and evaluate
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred).flatten()  # Ensure 1D array    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Number of parameters for Linear Regression
    num_params = X_train.shape[1] + 1  # Features + bias

    
    return model, y_pred, mse, r2, training_time, num_params
