from data_preprocessing import load_and_preprocess_data
from baseline_model import run_linear_regression
from mlp_baseline import run_mlp_baseline
from mlp_optimised import run_optimised_mlp
from utils import plot_training_histories, plot_actual_vs_predicted_all, plot_residuals_all, plot_combined_error_distribution, plot_residual_density
from mlp_optimised_gridsearch import run_gridsearch_mlp
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


# Load and preprocess data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Convert y_test to a NumPy array and flatten
y_test = y_test.values.flatten()
y_train = y_train.values.flatten()  # Ensure y_train is also 1D

# Run Linear Regression
lr_model, y_pred_lr_test, mse_lr_test, r2_lr_test, training_time_lr, num_params_lr = run_linear_regression(X_train, X_test, y_train, y_test)
y_pred_lr_train = lr_model.predict(X_train)  # Predict on training data
mse_lr_train = mean_squared_error(y_train, y_pred_lr_train)
r2_lr_train = r2_score(y_train, y_pred_lr_train)

# Run Baseline MLP
mlp_model, history_baseline, y_pred_mlp_test, mse_mlp_test, r2_mlp_test, training_time_mlp, num_params_mlp, epochs_mlp = run_mlp_baseline(X_train, X_test, y_train, y_test)
y_pred_mlp_train = mlp_model.predict(X_train).flatten()  # Predict on training data
mse_mlp_train = mean_squared_error(y_train, y_pred_mlp_train)
r2_mlp_train = r2_score(y_train, y_pred_mlp_train)

# Run Optimised MLP
opt_model, history_optimized, y_pred_opt_test, mse_opt_test, r2_opt_test, training_time_opt, num_params_opt, epochs_opt, best_params_opt = run_optimised_mlp(X_train, X_test, y_train, y_test)
y_pred_opt_train = opt_model.predict(X_train).flatten()  # Predict on training data
mse_opt_train = mean_squared_error(y_train, y_pred_opt_train)
r2_opt_train = r2_score(y_train, y_pred_opt_train)

# Run Optimised MLP
opt_gs_model, history_gs, y_pred_gs_test, best_params_gs, mse_gs_test, r2_gs_test, training_time_gs, num_params_gs = run_gridsearch_mlp(X_train, X_test, y_train, y_test)
y_pred_gs_train = opt_gs_model.predict(X_train).flatten()  # Predict on training data
mse_gs_train = mean_squared_error(y_train, y_pred_gs_train)
r2_gs_train = r2_score(y_train, y_pred_gs_train)

# GridSearch MLP
#best_params_gs, mse_gs, r2_gs, grid_search_time, num_params_gs = run_gridsearch_mlp(X_train, X_test, y_train, y_test)
#y_pred_opt_train = opt_model.predict(X_train).flatten()  # Predict on training data

# Compute residuals
residuals_lr = y_test - y_pred_lr_test
residuals_mlp = y_test - y_pred_mlp_test
residuals_opt = y_test - y_pred_opt_test
residuals_gs = y_test - y_pred_gs_test

# Combine residuals into a dictionary
residuals_dict = {
    "Linear Regression": residuals_lr,
    "Baseline MLP": residuals_mlp,
    "Keras Tuner MLP": residuals_opt,
    "GridSearch MLP": residuals_gs
}

# Print Results
print("\nModel Evaluation Results (Training and Test):")
print(f"{'Model':<20}{'Dataset':<10}{'MSE':<10}{'R²':<10}")
print(f"{'-'*50}")
print(f"{'Linear Regression':<20}{'Train':<10}{mse_lr_train:<10.4f}{r2_lr_train:<10.4f}")
print(f"{'Linear Regression':<20}{'Test':<10}{mse_lr_test:<10.4f}{r2_lr_test:<10.4f}")
print(f"{'Baseline MLP':<20}{'Train':<10}{mse_mlp_train:<10.4f}{r2_mlp_train:<10.4f}")
print(f"{'Baseline MLP':<20}{'Test':<10}{mse_mlp_test:<10.4f}{r2_mlp_test:<10.4f}")
print(f"{'Optimised MLP':<20}{'Train':<10}{mse_opt_train:<10.4f}{r2_opt_train:<10.4f}")
print(f"{'Optimised MLP':<20}{'Test':<10}{mse_opt_test:<10.4f}{r2_opt_test:<10.4f}")
print(f"{'GridSearch MLP':<20}{'Train':<10}{mse_gs_train:<10.4f}{r2_gs_train:<10.4f}")
print(f"{'GridSearch MLP':<20}{'Test':<10}{mse_gs_test:<10.4f}{r2_gs_test:<10.4f}")

# Plot training histories
plot_training_histories(history_baseline, history_optimized, history_gs)

# Plot actual vs. predicted values
plot_actual_vs_predicted_all(y_test, y_pred_lr_test, y_pred_mlp_test, y_pred_opt_test, y_pred_gs_test)

# Plot residuals
plot_residuals_all(y_test, y_pred_lr_test, y_pred_mlp_test, y_pred_opt_test, y_pred_gs_test)

# Plot combined residuals histogram
plot_combined_error_distribution(residuals_dict, title="Residuals Distribution for All Models")

plot_residual_density(residuals_dict, title="Residuals Density for All Models")

# Print computational efficiency metrics
print("\nComputational Efficiency Metrics:")
print(f"{'Model':<20}{'Training Time (s)':<20}{'Parameters':<15}{'Epochs to Converge':<20}")
print(f"{'-'*70}")
print(f"{'Linear Regression':<20}{training_time_lr:<20.2f}{num_params_lr:<15}")
print(f"{'Baseline MLP':<20}{training_time_mlp:<20.2f}{num_params_mlp:<15}")
print(f"{'Keras Tuner MLP':<20}{training_time_opt:<20.2f}{num_params_opt:<15}, Best Params: {best_params_opt}")
print(f"{'GridSearch MLP':<20}{training_time_gs:<20.2f}{num_params_gs:<15}, Best Params: {best_params_gs}")
#print(f"{'GridSearch MLP':<20}{mse_gs:<15.4f}{r2_gs:<10.4f}  Time: {grid_search_time:.2f}s, Params: {num_params_gs}, Best Params: {best_params_gs}")




