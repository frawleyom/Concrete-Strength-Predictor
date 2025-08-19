import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from keras import layers, optimizers
from scikeras.wrappers import KerasRegressor
import time
from sklearn.metrics import mean_squared_error, r2_score




# Function to build a fixed Keras model
def build_model(optimizer='adam', learning_rate=0.001, units_layer1=128, units_layer2=64, **kwargs):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(8,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    if optimizer == 'adam':
        opt = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model

def run_gridsearch_mlp(X_train, X_test, y_train, y_test):
    # Wrap the Keras model for GridSearchCV
    model = KerasRegressor(
        model=build_model,
        optimizer="adam",
        learning_rate=0.001,
        units_layer1=128,
        units_layer2=64,
        batch_size=32,
        epochs=50,
        verbose=0,
    )
    # Define hyperparameter grid
    param_grid = {
        'optimizer': ['adam', 'rmsprop'],
        'learning_rate': [0.001, 0.01],
        'units_layer1': [64, 128],
        'units_layer2': [32, 64],
        'batch_size': [16, 32],
        'epochs': [50]
    }

    # Grid search
    start_time = time.time()
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
    grid_result = grid.fit(X_train, y_train)
    end_time = time.time()
    grid_search_time = end_time - start_time

    # Best parameters
    best_params = grid_result.best_params_
    print("Best Parameters:", best_params)

    # Rebuild the best model with the best parameters
    #best_model = grid_result.best_estimator_
     # Rebuild the best model with the best parameters and retrain it to capture history
    best_model = build_model(
        optimizer=best_params['optimizer'],
        learning_rate=best_params['learning_rate']
    )   

    # Access the training history of the best model
    #history = best_model.history_  # Capture training history   
    history = best_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=0
    )
    

    y_pred = grid.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    num_params = best_model.count_params()  # Number of trainable parameters in the best model


    print(f"Test MSE: {mse:.4f}, Test R^2: {r2:.4f}")
    print(f"GridSearch Time: {grid_search_time:.2f}s")
    print(f"Trainable Parameters: {num_params}")

    return best_model, history, y_pred, best_params, mse, r2, grid_search_time, num_params
