import matplotlib.pyplot as plt
import numpy as np
import gzip

# One-hot encoding of the labels
def one_hot_encoding(label_data):
    # one-hot encoding
    encoded_labels = np.zeros((label_data.shape[0], 10)) # TODO: Write this function by yourself
    for i in range(label_data.shape[0]):
        encoded_labels[i, label_data[i]] = 1

    return encoded_labels

# Function to read pixel data from the dataset
def read_pixels(data_path):
    with gzip.open(data_path) as f:
        pixel_data = np.frombuffer(f.read(), 'B', offset=16).astype('float32')
    normalized_pixels = pixel_data / 255
    data_size = normalized_pixels.shape[0]
    image_count = int(data_size / 784)
    
    flattened_pixels = pixel_data.reshape(image_count, 784) # TODO: Flatten the normalized pixels
    return flattened_pixels

# Function to read label data from the dataset
def read_labels(data_path):
    with gzip.open(data_path) as f:
        label_data = np.frombuffer(f.read(), 'B', offset=8)
    one_hot_encoding_labels = one_hot_encoding(label_data)
    return one_hot_encoding_labels

# Function to read the entire dataset
def read_dataset():
    X_train = read_pixels(r"data\train-images-idx3-ubyte.gz")
    y_train = read_labels(r"data\train-labels-idx1-ubyte.gz")
    X_test = read_pixels(r"data\t10k-images-idx3-ubyte.gz")
    y_test = read_labels(r"data\t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    # For Question 2.4
    # Code to visualize weights (use your own weight variable, adjust its shape by yourself)
    # plt.matshow(weight, cmap=plt.cm.gray, vmin=0.5*weight.min(), vmax=0.5*weight.max())

    x_train, y_train, x_test, y_test = read_dataset()

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

