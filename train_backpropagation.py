import numpy as np
import matplotlib.pyplot as plt

class NeuralNet:
    def __init__(self, layer_dims, lr=0.01, max_epochs=10000, tolerance=0.001, momentum=0.9, mode='1'):
        self.layer_dims = layer_dims
        self.lr = lr
        self.max_epochs = max_epochs
        self.tolerance = tolerance
        self.momentum = momentum
        self.params = self.init_params()
        self.velocities = self.init_velocities()
        self.loss_history = []
        self.mode = mode
        self.accuracy = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_grad(self, a):
        return a * (1 - a)

    def identity(self, z):
        return z

    def identity_grad(self, a):
        return np.ones_like(a)
    
    def init_params(self):
        parameters = []
        for i in range(len(self.layer_dims) - 1):
            W = np.random.randn(self.layer_dims[i], self.layer_dims[i + 1])
            b = np.zeros((1, self.layer_dims[i + 1]))
            parameters.append((W, b))
        return parameters

    def init_velocities(self):
        velocities = []
        for i in range(len(self.layer_dims) - 1):
            vW = np.zeros((self.layer_dims[i], self.layer_dims[i + 1]))
            vb = np.zeros((1, self.layer_dims[i + 1])) 
            velocities.append((vW, vb))
        return velocities

    def forward(self, X):
        A = X
        cache = []
        for i in range(len(self.params) - 1):
            W, b = self.params[i]
            Z = np.dot(A, W) + b
            A = self.sigmoid(Z)
            cache.append((A, Z))
        W, b = self.params[-1]
        Z = np.dot(A, W) + b
        A = self.sigmoid(Z)
        cache.append((A, Z))
        return cache

    def backward(self, X, Y, cache):
        m = Y.shape[0]
        grads = []
        A, Z = cache[-1]
        dZ = A - Y
        dW = (1/m) * np.dot(cache[-2][0].T, dZ) if len(cache) > 1 else (1/m) * np.dot(X.T, dZ)
        db = (1/m) * np.sum(dZ, axis=0)
        grads.append((dW, db))
        
        for i in range(len(cache) - 2, -1, -1):
            A, Z = cache[i]
            dA = np.dot(dZ, self.params[i + 1][0].T)
            dZ = dA * self.sigmoid_grad(A)
            dW = (1/m) * np.dot(cache[i - 1][0].T, dZ) if i > 0 else (1/m) * np.dot(X.T, dZ)
            db = (1/m) * np.sum(dZ, axis=0)
            grads.append((dW, db))
        
        grads.reverse()
        
        for i in range(len(self.params)):
            W, b = self.params[i]
            dW, db = grads[i]
            vW, vb = self.velocities[i]
            
            vW = self.momentum * vW + (1 - self.momentum) * dW
            vb = self.momentum * vb + (1 - self.momentum) * db

            W -= self.lr * vW
            b -= self.lr * vb

            self.params[i] = (W, b)
            self.velocities[i] = (vW, vb)

    def normalize_data(self, X, Y):
        epsilon = 1e-8
        X_normalized = (X - np.min(X)) / (np.max(X) - np.min(X) + epsilon)
        Y_normalized = (Y - np.min(Y)) / (np.max(Y) - np.min(Y) + epsilon)
        return X_normalized, Y_normalized

    def fit(self, X, Y):
        if self.mode == '1':
            X, Y = self.normalize_data(X, Y)

        for epoch in range(self.max_epochs):
            cache = self.forward(X)
            self.backward(X, Y, cache)
            
            mse_loss = np.mean((Y - cache[-1][0]) ** 2)
            print(f'Epoch {epoch}, Loss: {mse_loss}')

            if mse_loss <= 100:
                self.loss_history.append(mse_loss)

            if mse_loss <= self.tolerance:
                break

    def evaluate(self, X_test, Y_test):
        if self.mode == '1':
            X_test_norm, Y_test_norm = self.normalize_data(X_test, Y_test)
            preds = self.predict(X_test_norm)
            return Y_test_norm, preds
        elif self.mode == '2':
            preds = self.predict(X_test)
            preds = np.round(preds)
            return Y_test, preds
            
    def predict(self, X):
        Y_dummy = 0
        X_norm, Y_dummy = self.normalize_data(X, Y_dummy)
        cache = self.forward(X_norm)
        return cache[-1][0]
    
    def plot_confusion_matrix(self, all_targets, all_preds):
        if self.mode == '2':
            combined_matrix = np.zeros((2, 2), dtype=int)
            
            for target, pred in zip(all_targets, all_preds):
                y_true = np.argmax(target, axis=1)
                y_pred = np.argmax(pred, axis=1)
                confusion = np.zeros((2, 2), dtype=int)
                for true, predicted in zip(y_true, y_pred):
                    confusion[true, predicted] += 1
                combined_matrix += confusion
            
            TP = combined_matrix[0, 0]
            FP = combined_matrix[0, 1]
            FN = combined_matrix[1, 0]
            TN = combined_matrix[1, 1]
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            self.accuracy = accuracy * 100

            fig, ax = plt.subplots()
            cax = ax.matshow(combined_matrix, cmap=plt.cm.Blues)
            plt.colorbar(cax)

            for (i, j), val in np.ndenumerate(combined_matrix):
                plt.text(j, i, val, ha='center', va='center', color='red')

            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title('Confusion Matrix')

            plt.xticks([0, 1], ['[1,0]', '[0,1]'])
            plt.yticks([0, 1], ['[1,0]', '[0,1]'])

def load_dataset(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    data_lines = lines[2:]
    data = []

    for line in data_lines:
        data.append([int(value) for value in line.split()])

    return np.array(data)

def load_cross_data(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    data = []
    current_in = []
    current_out = []
    for line in lines:
        line = line.strip()
        if line.startswith('p'):
            if current_in and current_out:
                combined = current_in + current_out
                data.append(combined)
                current_in = []
                current_out = []
        else:
            if not current_in:
                current_in = [float(num) for num in line.split()]
            else:
                current_out = [int(num) for num in line.split()]

    if current_in and current_out:
        combined = current_in + current_out
        data.append(combined)
    
    return data

def shuffle_split(data):
    np.random.shuffle(data)
    X = np.array(data[2:, :-1])
    y = np.array(data[2:, -1])
    Y = [[i] for i in y]
    return X, np.array(Y)

def shuffle_split_cross(c_data):
    c_data = np.array(c_data)
    np.random.shuffle(c_data)
    Xc = np.array(c_data[:, :int(len(c_data[0]) / 2)])
    Yc = np.array(c_data[:, -int(len(c_data[0]) / 2):])
    return Xc, Yc

def k_fold_split(X, Y, k, fold):
    X_split = np.array_split(X, k)
    Y_split = np.array_split(Y, k)
    X_test = X_split[fold]
    Y_test = Y_split[fold]
    X_train = np.concatenate([X_split[j] for j in range(k) if j != fold])
    Y_train = np.concatenate([Y_split[j] for j in range(k) if j != fold])
    return X_train, Y_train, X_test, Y_test

if __name__ == '__main__':
    mode = input("Choose data type 1.Regression 2.Classification :")
    
    file1 = 'dataset.txt'
    file2 = 'cross.txt'
    dataset = load_dataset(file1)
    cross_data = load_cross_data(file2)
    X, Y = shuffle_split(dataset)
    Xc, Yc = shuffle_split_cross(cross_data)

    # Hyperparameters
    reg_layers = [8, 16, 1]
    clf_layers = [2, 16, 2]
    max_epochs = 10000
    lr = 0.9
    tolerance = 0.001
    momentum = 0.95

    k = 10
    if mode == '1':
        all_losses = []
        for i in range(k):
            nn = NeuralNet(reg_layers, lr, max_epochs, tolerance, momentum, mode)
            X_train, Y_train, X_test, Y_test = k_fold_split(X, Y, k, i)
            nn.loss_history = []
            nn.fit(X_train, Y_train)
            Y_norm, predictions = nn.evaluate(X_test, Y_test)
            all_losses.append(nn.loss_history)
        
        plt.figure()
        for i, losses in enumerate(all_losses):
            plt.plot(losses, label=f'Fold {i+1}')
        plt.title('MSE vs. Epoch for All Folds')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid()
        
        plt.figure(figsize=(12, 12))  

        for i, losses in enumerate(all_losses):
            plt.subplot(5, 2, i + 1)
            plt.plot(losses, label=f'Fold {i+1}')
            plt.title(f'Fold {i+1} - MSE vs. Epoch')
            plt.xlabel('Epoch')
            plt.ylabel('MSE')
            plt.legend()
            plt.grid()

        plt.tight_layout()
        plt.show()

    elif mode == '2':
        all_targets = []
        all_preds = []
        for i in range(k):
            clf = NeuralNet(clf_layers, 0.85, max_epochs, tolerance, 0.9, mode)
            Xc_train, Yc_train, Xc_test, Yc_test = k_fold_split(Xc, Yc, k, i)
            clf.loss_history = []
            clf.fit(Xc_train, Yc_train)
            Y_test, predictions = clf.evaluate(Xc_test, Yc_test)
            all_targets.append(Y_test)
            all_preds.append(predictions)
        
        clf.plot_confusion_matrix(all_targets, all_preds)
        print('___________________________________________')
        print(f"Accuracy rate across folds: {clf.accuracy:.2f} %")

    else:
        raise ValueError("Input should be 1 or 2!")

    plt.show()
