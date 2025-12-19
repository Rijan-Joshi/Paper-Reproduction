# Here, we load MNIST dataset and preprocess it
from src.utils.cupy_numpy import np
from tensorflow.keras.datasets import mnist


def pad(arr, a, b):
    return np.pad(arr, ((0, 0), (a, b), (b, b)), mode="constant")


def load_data():

    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.astype("float32")
    y_train = y_train.astype("int32")
    X_test = X_test.astype("float32")
    y_test = y_test.astype("int32")

    # Normalization on the  basis of LeNet-5
    X_train = -0.005 * X_train + 1.175
    X_test = -0.005 * X_test + 1.175

    # Since LeNet uses  32 x 32 image, we need to pad the image
    X_train = pad(X_train, 2, 2)
    X_test = pad(X_test, 2, 2)

    # Add channel dimension
    X_train = np.array(X_train[:, np.newaxis, :, :])
    X_test = np.array(X_test[:, np.newaxis, :, :])

    return (X_train, X_test, y_train, y_test)


load_data()


# URL was not working

# def download_mnist(data_dir="data"):
#     """
#     Downlaod the MNIST data
#     """

#     os.makedirs(data_dir, exist_ok=True)

#     base_url = "http://yann.lecun.com/exdb/mnist/"
#     files = [
#         "train-images-idx3-ubyte.gz",
#         "train-labels-idx1-ubyte.gz",
#         "t10k-images-idx3-ubyte.gz",
#         "t10k-labels-idx1-ubyte.gz",
#     ]

#     for file in files:
#         filepath = os.path.join(data_dir, file)
#         url = base_url + file
#         if not os.path.exists(filepath):
#             urllib.request.urlretrieve(url, filepath)

#     print("MNIST dataset downloaded")


# download_mnist()
