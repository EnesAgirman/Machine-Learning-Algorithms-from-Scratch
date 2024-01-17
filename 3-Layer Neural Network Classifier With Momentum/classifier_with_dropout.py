import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class ThreeLayerNet:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size
        self.initialize_parameters(input_size, hidden1_size, hidden2_size, output_size)
        self.validation_losses = []

    def initialize_parameters(self, input_size, hidden1_size, hidden2_size, output_size):
        np.random.seed(41) # 42nd number starting from 0: the answer to the ultimate question of life, the universe and everything
        self.w1 = np.random.uniform(-0.1, 0.1, (input_size, hidden1_size))
        self.w2 = np.random.uniform(-0.1, 0.1, (hidden1_size, hidden2_size))
        self.w3 = np.random.uniform(-0.1, 0.1, (hidden2_size, output_size))
        
        self.b1 = np.zeros((1, hidden1_size))
        self.b2 = np.zeros((1, hidden2_size))
        self.b3 = np.zeros((1, output_size))
        
        self.dw1 = np.zeros_like(self.w1)
        self.dw2 = np.zeros_like(self.w2)
        self.dw3 = np.zeros_like(self.w3)
        
        self.db1 = np.zeros_like(self.b1)
        self.db2 = np.zeros_like(self.b2)
        self.db3 = np.zeros_like(self.b3)

        return self.w1, self.b1, self.w2, self.b2, self.w3, self.b3

    def relu(self, X):
        return np.maximum(0, X)
    
    def relu_derivative(self, X):
        return (X > 0).astype(int)

    def softmax(self, X):
        X = np.clip(X, -500, 500) # clip z to avoid overflow
        expX = np.exp(X - np.max(X, axis=1, keepdims=True))
        result = expX / np.sum(expX, axis=1, keepdims=True)
        return result
    
    def softmax_derivative(self, X):
        return X * (1 - X)

    def forward(self, X, w1, b1, w2, b2, w3, b3, dropout_rate=0.5):
        self.z1 = np.dot(X, w1) + b1
        self.h1 = self.relu(self.z1)

        # Layer 2 with Dropout
        self.z2 = np.dot(self.h1, w2) + b2
        self.h2 = self.relu(self.z2)
        self.d2 = np.random.rand(*self.h2.shape) < dropout_rate
        self.h2 *= self.d2
        self.h2 /= dropout_rate        
        
        
        self.z3 = np.dot(self.h2, w3) + b3
        self.h3 = self.softmax(self.z3)

        return self.h3

    def backward_propagation(self, X, Y, W2, W3, dropout_rate=0.5):

        dZ3 = self.h3 - Y
        dW3 = np.dot(self.h2.T, dZ3)
        db3 = np.sum(dZ3, axis=0, keepdims=True) / X.shape[0]

        # Layer 2 with Dropout
        dA2 = np.dot(dZ3, W3.T)
        dA2 *= self.d2
        dA2 /= dropout_rate
        dZ2 = dA2 * self.relu_derivative(self.z2)
        dW2 = np.dot(self.h1.T, dZ2)
        db2 = np.sum(dZ2, axis=0, keepdims=True) / X.shape[0]
        
        dA1 = np.dot(dZ2, W2.T)
        dZ1 = dA1 * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dZ1)
        db1 = np.sum(dZ1, axis=0, keepdims=True) / X.shape[0]
        
        # clip gradients to avoid exploding gradients
        dW1 = np.clip(dW1, -2, 2)
        dW2 = np.clip(dW2, -2, 2)
        dW3 = np.clip(dW3, -2, 2)

        return dW1, db1, dW2, db2, dW3, db3

    def update_parameters(self, W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate, momentum_coefficient):
        
        self.dw1 = momentum_coefficient * self.dw1 + learning_rate * dW1
        self.db1 = momentum_coefficient * self.db1 + learning_rate * db1
        self.dw2 = momentum_coefficient * self.dw2 + learning_rate * dW2
        self.db2 = momentum_coefficient * self.db2 + learning_rate * db2
        self.dw3 = momentum_coefficient * self.dw3 + learning_rate * dW3
        self.db3 = momentum_coefficient * self.db3 + learning_rate * db3
        
        self.w1 -= self.dw1
        self.b1 -= self.db1
        self.w2 -= self.dw2
        self.b2 -= self.db2
        self.w3 -= self.dw3
        self.b3 -= self.db3

        return W1, b1, W2, b2, W3, b3

    def train_model(self, X_train, Y_train, X_val, Y_val, W1, b1, W2, b2, W3, b3, epochs, batch_size, learning_rate, momentum_coefficient):
        
        if batch_size == 0:
            batch_size = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle the training data
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train_shuffled = X_train[indices]
            Y_train_shuffled = Y_train[indices]
            
            epoch_accuracy = 0
            epoch_loss = 0

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train_shuffled[i:i + batch_size]
                Y_batch = Y_train_shuffled[i:i + batch_size]

                output = self.forward(X_batch, W1, b1, W2, b2, W3, b3)

                dW1, db1, dW2, db2, dW3, db3 = self.backward_propagation(X_batch, Y_batch, W2, W3)

                W1, b1, W2, b2, W3, b3 = self.update_parameters(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, learning_rate, momentum_coefficient)
                
                # Compute loss and accuracy
                loss = self.compute_loss(Y_batch, output)
                epoch_loss += loss * X_batch.shape[0]
                epoch_accuracy += self.compute_accuracy(Y_batch, output) * X_batch.shape[0]

            # calculate and display the validation loss and accuracy for each epoch
            val_output = self.forward(X_val, W1, b1, W2, b2, W3, b3)
            val_loss = self.compute_loss(Y_val, val_output)
            val_accuracy = self.compute_accuracy(Y_val, val_output)
            print(f'Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            self.validation_losses.append(val_loss)

    def compute_loss(self, Y, Y_hat):
        # cross entropy loss
        return -np.sum(Y * np.log(Y_hat + 1e-8) + (1 - Y) * np.log(1 - Y_hat + 1e-8)) / Y.shape[0]

    def compute_accuracy(self, Y_true, Y_pred):
        predictions = np.argmax(Y_pred, axis=1)
        labels = np.argmax(Y_true, axis=1)
        return np.mean(predictions == labels)
    
    def test_model(self, X_test, Y_test, W1, b1, W2, b2, W3, b3):
        pred = self.forward(X_test, W1, b1, W2, b2, W3, b3)
        test_loss = self.compute_loss(Y_test, pred)
        test_accuracy = self.compute_accuracy(Y_test, pred)
        return test_loss, test_accuracy

def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels - 1]

if __name__ == "__main__":
    
    # hyperparameters
    num_classes = 6
    input_size = 561  # Number of input features
    hidden1_size = 300
    hidden2_size = 200
    output_size = num_classes   # one output neuron for each class
    learning_rate = 0.01
    momentum_coefficient = 0
    epochs = 100
    batch_size = 50
    # Note: I didn't understand what do you mean by making the batch size 0. Right now, I am using the whole training
    # set as one batch when batch_size = 0. If you meant for it to be batch size = 1, then you can just give it 1 as the batch size.
    validation_split = 0.1
    
    # load data
    x_train_path = r"UCI HAR Dataset\train\X_train.txt"
    y_train_path = r"UCI HAR Dataset\train\y_train.txt"
    x_test_path = r"UCI HAR Dataset\test\X_test.txt"
    y_test_path = r"UCI HAR Dataset\test\y_test.txt"
    
    # read the data as pandas dataframes
    x_train = pd.read_csv(x_train_path, header=None, delim_whitespace=True).values
    y_train = pd.read_csv(y_train_path, header=None, delim_whitespace=True).values.flatten()
    x_test = pd.read_csv(x_test_path, header=None, delim_whitespace=True).values
    y_test = pd.read_csv(y_test_path, header=None, delim_whitespace=True).values.flatten()

    # Normalize the data
    X_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
    X_test = (x_test - np.mean(x_test, axis=0)) / np.std(x_test, axis=0)

    # One-hot encode the labels
    y_train_encoded = one_hot_encode(y_train, num_classes)
    y_test_encoded = one_hot_encode(y_test, num_classes)

    # Split training validation set
    num_validation_samples = int(validation_split * X_train.shape[0])

    # seperate the validation set from the training set
    X_val = X_train[:num_validation_samples]
    y_val = y_train_encoded[:num_validation_samples]
    X_train = X_train[num_validation_samples:]
    y_train_encoded = y_train_encoded[num_validation_samples:]

    # initialize model
    myModel = ThreeLayerNet(input_size, hidden1_size, hidden2_size, output_size)

    # Train the model
    myModel.train_model(X_train, y_train_encoded, X_val, y_val, myModel.w1, myModel.b1, myModel.w2, myModel.b2, myModel.w3, myModel.b3, epochs, batch_size, learning_rate, momentum_coefficient)

    # Plot the validation losses by epochs
    plt.plot(myModel.validation_losses)
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss by Epochs')
    plt.show()
    
    
    # Test the model
    test_loss, test_accuracy = myModel.test_model(X_test, y_test_encoded, myModel.w1, myModel.b1, myModel.w2, myModel.b2, myModel.w3, myModel.b3)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
