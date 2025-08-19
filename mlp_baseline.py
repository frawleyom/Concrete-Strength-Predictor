import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from sklearn.metrics import mean_squared_error, r2_score
import time

def run_mlp_baseline(X_train, X_test, y_train, y_test):
    # Define the model
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    num_params = model.count_params()


    # Start timing
    start_time = time.time()
    
    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time

    # Predict and evaluate
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    epochs_to_converge = len(history.history['loss'])  # Total epochs run

    
    return model, history, y_pred, mse, r2, training_time, num_params, epochs_to_converge
