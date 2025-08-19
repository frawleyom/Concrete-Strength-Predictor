import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
import keras_tuner as kt
from tensorflow import keras
from keras import layers, optimizers
from sklearn.metrics import mean_squared_error, r2_score
import time

def build_model(hp):
    """
    Builds and returns a Keras model with hyperparameters for tuning.

    Args:
        hp: Hyperparameters object provided by Keras Tuner.

    Returns:
        Compiled Keras model.
    """
    model = keras.Sequential()

    # Input layer with tunable units and activation
    model.add(layers.Dense(
        hp.Int('units_input', min_value=32, max_value=256, step=32),  # Units in input layer
        activation=hp.Choice('activation_input', ['relu', 'tanh', 'sigmoid']),  # Activation function
        input_shape=(8,)  # Assuming 8 features in the input
    ))
    
    # Hidden layers with tunable number of layers
    for i in range(hp.Int('num_layers', min_value=1, max_value=5)):  # Up to 5 hidden layers
        model.add(layers.Dense(
            hp.Int(f'units_layer_{i}', min_value=32, max_value=256, step=32),  # Units in each layer
            activation=hp.Choice(f'activation_layer_{i}', ['relu', 'tanh', 'sigmoid'])  # Activation function
        ))

        # Optional dropout layer
        if hp.Boolean(f'use_dropout_layer_{i}'):  # Add dropout layer if chosen
            model.add(layers.Dropout(
                hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1)  # Dropout rate
            ))

    # Output layer
    model.add(layers.Dense(1))  # Single output for regression

    # Optimizer with tunable learning rate and type
    optimizer_name = hp.Choice('optimizer', ['adam', 'rmsprop'])  # Choice of optimizer
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')  # Tunable learning rate

    if optimizer_name == 'adam':
        optimizer = optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name == 'rmsprop':
        optimizer = optimizers.RMSprop(learning_rate=learning_rate)

    # Compile the model
    model.compile(
        optimizer=optimizer,
        loss='mse',  # Mean Squared Error for regression
        metrics=['mae']  # Mean Absolute Error for monitoring
    )

    # Add tunable batch size (not used here directly but accessible in training function)
    hp.Choice('batch_size', [16, 32, 64, 128])  # Batch sizes for training

    return model




def run_optimised_mlp(X_train, X_test, y_train, y_test):

    total_start_time = time.time()

    tuner = kt.Hyperband(
        build_model,
        objective='val_loss',
        max_epochs=50,
        factor=3,
        directory='tuner_logs',
        project_name='mlp_optimization_expanded'
    )
    tuner.search(X_train, y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

    # Retrieve best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Debug best hyperparameters
    print("Best Hyperparameters:")
    for hp_name, hp_value in best_hps.values.items():
        print(f"{hp_name}: {hp_value}")

    # Build model with best hyperparameters
    model = tuner.hypermodel.build(best_hps)
    num_params = model.count_params()

    # Ensure y_train is flattened
    y_train = y_train.flatten()

    # Start timing
    start_time = time.time()
    
    # Check and use batch size from hyperparameters
    batch_size = best_hps.get('batch_size')
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=50, batch_size=batch_size, verbose=0)

    end_time = time.time()
    final_training_time = end_time - start_time
    
    # End total timing
    total_end_time = time.time()
    total_time = total_end_time - total_start_time

    # Predict and evaluate
    y_pred = model.predict(X_test).flatten()
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    epochs_to_converge = len(history.history['loss'])  # Total epochs run
    
    return model, history, y_pred, mse, r2, total_time, num_params, epochs_to_converge, best_hps

