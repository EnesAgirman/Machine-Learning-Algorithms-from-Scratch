import numpy as np
import gzip
import matplotlib.pyplot as plt

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
    return label_data

def display_image(image, label):
    plt.imshow(image.reshape(28, 28), cmap='gray')
    plt.title("Label: " + str(label))
    plt.show()

if __name__ == "__main__":
    images = read_pixels("data/data/train-images-idx3-ubyte.gz")
    labels = read_labels("data/data/train-labels-idx1-ubyte.gz")

    test_images = read_pixels("data/data/t10k-images-idx3-ubyte.gz")
    test_labels = read_labels("data/data/t10k-labels-idx1-ubyte.gz")

    print(test_images.shape)
    print(test_labels.shape)

    # display a random image with its label
    num = np.random.randint(0, 10000)
    display_image(test_images[num], test_labels[num])