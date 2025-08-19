import matplotlib.pyplot as plt

def plot_training_histories(history_baseline, history_optimized, history_gs):
    """Plot training and testing loss for both MLP models."""
    plt.figure(figsize=(8, 6))

    # Baseline MLP
    plt.plot(history_baseline.history['loss'], label='Baseline Training Loss', linestyle='-')
    plt.plot(history_baseline.history['val_loss'], label='Baseline Testing Loss', linestyle='--')

    # Optimized MLP
    plt.plot(history_optimized.history['loss'], label='Keras Tuner Training Loss', linestyle='-')
    plt.plot(history_optimized.history['val_loss'], label='Keras Tuner Testing Loss', linestyle='--')

    plt.plot(history_gs.history['loss'], label='GridSearch Training Loss', linestyle='-')
    plt.plot(history_gs.history['val_loss'], label='GridSearch Testing Loss', linestyle='--')

    plt.title("Training and Testing Loss for MLP Models", fontsize=16)
    plt.xlabel("Epochs", fontsize=14)
    plt.ylabel("Loss (MSE)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

def plot_actual_vs_predicted_all(y_true, y_pred_lr, y_pred_mlp, y_pred_opt, y_pred_gs):
    """Plot actual vs. predicted values for all three models."""
    plt.figure(figsize=(8, 6))

    plt.scatter(y_true, y_pred_lr, alpha=0.7, label='Linear Regression')
    plt.scatter(y_true, y_pred_mlp, alpha=0.7, label='MLP Baseline')
    plt.scatter(y_true, y_pred_opt, alpha=0.7, label='MLP Keras Tuner')
    plt.scatter(y_true, y_pred_gs, alpha=0.7, label='MLP GridSearch')

    min_val, max_val = min(y_true), max(y_true)
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

    plt.title("Actual vs. Predicted Values", fontsize=16)
    plt.xlabel("Actual Values", fontsize=14)
    plt.ylabel("Predicted Values", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import numpy as np

def plot_residuals_all(y_true, y_pred_lr, y_pred_mlp, y_pred_opt, y_pred_gs):
    """Plot residuals (errors) for all three models."""
    # Flatten all inputs
    y_true = np.array(y_true).flatten()
    y_pred_lr = np.array(y_pred_lr).flatten()
    y_pred_mlp = np.array(y_pred_mlp).flatten()
    y_pred_opt = np.array(y_pred_opt).flatten()
    y_pred_gs = np.array(y_pred_gs).flatten()

    # Calculate residuals
    residuals_lr = y_true - y_pred_lr
    residuals_mlp = y_true - y_pred_mlp
    residuals_opt = y_true - y_pred_opt
    residuals_gs = y_true - y_pred_gs

    # Plot residuals
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, residuals_lr, alpha=0.7, label='Linear Regression')
    plt.scatter(y_true, residuals_mlp, alpha=0.7, label='MLP Baseline')
    plt.scatter(y_true, residuals_opt, alpha=0.7, label='MLP Keras Tuner')
    plt.scatter(y_true, residuals_gs, alpha=0.7, label='MLP GridSearch')

    plt.axhline(0, color='red', linestyle='--', label='Zero Error')
    plt.title("Residuals for All Models", fontsize=16)
    plt.xlabel("Actual Values", fontsize=14)
    plt.ylabel("Residuals (Error)", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt

def plot_combined_error_distribution(residuals_dict, title="Residuals Distribution for All Models"):
    """
    Plots a combined histogram of residuals for multiple models.
    
    Args:
        residuals_dict (dict): A dictionary where keys are model names and values are residual arrays.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))

    # Plot each model's residuals on the same histogram
    for model_name, residuals in residuals_dict.items():
        plt.hist(residuals, bins=30, alpha=0.5, label=model_name, edgecolor='black')

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel("Residuals", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_residual_density(residuals_dict, title="Residuals Density for All Models"):
    """
    Plots the density of residuals for multiple models using KDE.
    
    Args:
        residuals_dict (dict): A dictionary where keys are model names and values are residual arrays.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(8, 6))

    # Plot each model's residual density
    for model_name, residuals in residuals_dict.items():
        sns.kdeplot(residuals, label=model_name, fill=True, alpha=0.3)

    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel("Residuals", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()

