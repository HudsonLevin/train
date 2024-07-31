import numpy as np
import matplotlib.pyplot as plt

# Function to load and normalize dataset
def load_and_normalize_dataset(file_path):
    dataset = np.loadtxt(file_path)
    features = dataset[:, :-1]
    targets = dataset[:, -1].reshape(-1, 1)
    
    mean = np.mean(features, axis=0)
    std_dev = np.std(features, axis=0)
    normalized_features = (features - mean) / std_dev
    
    return normalized_features, targets, mean, std_dev

# Function to initialize network parameters
def initialize_network_params(input_size, hidden_sizes, output_size):
    params = {}
    layer_sizes = [input_size] + hidden_sizes + [output_size]
    
    for i in range(1, len(layer_sizes)):
        params[f"W{i}"] = np.random.randn(layer_sizes[i-1], layer_sizes[i]) * 0.01
        params[f"b{i}"] = np.zeros((1, layer_sizes[i]))
    
    return params

# Function to initialize momentum terms
def initialize_momentum_terms(params):
    momentum_terms = {}
    total_layers = len(params) // 2
    for i in range(1, total_layers + 1):
        momentum_terms[f"dW{i}"] = np.zeros_like(params[f"W{i}"])
        momentum_terms[f"db{i}"] = np.zeros_like(params[f"b{i}"])
    return momentum_terms

# Sigmoid activation function
def sigmoid_activation(Z):
    return 1 / (1 + np.exp(-Z))

# Forward pass through the network
def forward_pass(X, params, hidden_sizes):
    cache = {}
    A = X
    total_layers = len(hidden_sizes) + 1
    
    for i in range(1, total_layers):
        Z = np.dot(A, params[f"W{i}"]) + params[f"b{i}"]
        A = sigmoid_activation(Z)
        cache[f"Z{i}"] = Z
        cache[f"A{i}"] = A
    
    Z_final = np.dot(A, params[f"W{total_layers}"]) + params[f"b{total_layers}"]
    A_final = Z_final
    cache[f"Z{total_layers}"] = Z_final
    cache[f"A{total_layers}"] = A_final
    
    return A_final, cache

# Mean squared error loss
def compute_mse_loss(Y_true, Y_pred):
    return np.mean((Y_true - Y_pred) ** 2)

# Calculate percentage error
def compute_percentage_error(Y_true, Y_pred):
    return np.mean(np.abs((Y_true - Y_pred) / Y_true)) * 100

# Derivative of sigmoid function
def sigmoid_derivative(Z):
    s = sigmoid_activation(Z)
    return s * (1 - s)

# Backward pass to compute gradients
def backward_pass(X, Y, params, cache, hidden_sizes):
    gradients = {}
    m = X.shape[0]
    total_layers = len(hidden_sizes) + 1
    
    dZ_final = 2 * (cache[f"A{total_layers}"] - Y)
    gradients[f"dW{total_layers}"] = np.dot(cache[f"A{total_layers-1}"].T, dZ_final) / m
    gradients[f"db{total_layers}"] = np.sum(dZ_final, axis=0, keepdims=True) / m
    
    for i in reversed(range(1, total_layers)):
        dA_prev = np.dot(dZ_final, params[f"W{i+1}"].T)
        dZ = dA_prev * sigmoid_derivative(cache[f"Z{i}"])
        gradients[f"dW{i}"] = np.dot(X.T, dZ) / m if i == 1 else np.dot(cache[f"A{i-1}"].T, dZ) / m
        gradients[f"db{i}"] = np.sum(dZ, axis=0, keepdims=True) / m
        dZ_final = dZ
    
    return gradients

# Update network parameters using momentum
def update_params_with_momentum(params, gradients, momentum_terms, learning_rate, momentum):
    total_layers = len(params) // 2
    
    for i in range(1, total_layers + 1):
        momentum_terms[f"dW{i}"] = momentum * momentum_terms[f"dW{i}"] + (1 - momentum) * gradients[f"dW{i}"]
        momentum_terms[f"db{i}"] = momentum * momentum_terms[f"db{i}"] + (1 - momentum) * gradients[f"db{i}"]
        
        params[f"W{i}"] -= learning_rate * momentum_terms[f"dW{i}"]
        params[f"b{i}"] -= learning_rate * momentum_terms[f"db{i}"]
    
    return params, momentum_terms

# Training the neural network
def train_neural_network(X, Y, hidden_sizes, epochs, learning_rate, momentum):
    input_size = X.shape[1]
    output_size = Y.shape[1]
    params = initialize_network_params(input_size, hidden_sizes, output_size)
    momentum_terms = initialize_momentum_terms(params)
    
    loss_history = []
    percent_error_history = []
    
    for epoch in range(epochs):
        Y_pred, cache = forward_pass(X, params, hidden_sizes)
        loss = compute_mse_loss(Y, Y_pred)
        percent_error = compute_percentage_error(Y, Y_pred)
        gradients = backward_pass(X, Y, params, cache, hidden_sizes)
        params, momentum_terms = update_params_with_momentum(params, gradients, momentum_terms, learning_rate, momentum)
        
        loss_history.append(loss)
        percent_error_history.append(percent_error)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, MSE Loss: {loss:.4f}, Percent Error: {percent_error:.2f}%")
    
    return params, loss_history, percent_error_history

# Load and normalize the dataset
file_path = 'dataset.txt'
X, Y, mean, std_dev = load_and_normalize_dataset(file_path)

# Split dataset into training and testing sets
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
Y_train, Y_test = Y[:train_size], Y[train_size:]

# Train the neural network
hidden_sizes = [10, 5]
epochs = 15000
learning_rate = 0.0001
momentum = 0.9
params, loss_history, percent_error_history = train_neural_network(X_train, Y_train, hidden_sizes, epochs, learning_rate, momentum)

# Make predictions on training and testing sets
Y_train_pred, _ = forward_pass(X_train, params, hidden_sizes)
Y_test_pred, _ = forward_pass(X_test, params, hidden_sizes)

# Print prediction results
print("Training Predictions:")
print(Y_train_pred[:10])  # Print first 10 predictions for brevity
print("Testing Predictions:")
print(Y_test_pred[:10])   # Print first 10 predictions for brevity

# Save predictions to files
np.savetxt('train_predictions.txt', Y_train_pred)
np.savetxt('test_predictions.txt', Y_test_pred)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.plot(range(len(Y_train)), Y_train, label='Actual (Train)', alpha=0.6)
plt.plot(range(len(Y_train)), Y_train_pred, label='Predicted (Train)', alpha=0.7)
plt.title('Training Data')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(range(len(Y_test)), Y_test, label='Actual (Test)', alpha=0.6)
plt.plot(range(len(Y_test)), Y_test_pred, label='Predicted (Test)', alpha=0.7)
plt.title('Testing Data')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(range(epochs), loss_history, label='MSE Loss', alpha=0.7)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(range(epochs), percent_error_history, label='Percent Error', alpha=0.7)
plt.title('Training Percent Error')
plt.xlabel('Epoch')
plt.ylabel('Percent Error')
plt.legend()

plt.tight_layout()
plt.show()
