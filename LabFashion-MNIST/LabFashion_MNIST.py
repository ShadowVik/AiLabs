import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Load the Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Normalize the pixel values
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255

# Add an extra dimension for the channel
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode the labels
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define the CNN modelkernel_size=()
model = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    #In summary, the MaxPooling2D layer in a CNN reduces the spatial dimensions of the input feature maps and 
    #extracts the most important features, while also helping to prevent overfitting and make the model more robust 
    #to small translations in the input image. 16 to 4
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    #The Flatten layer simply flattens the 7x7x64 tensor into a one-dimensional vector of length 3136. 
    #The output shape of the Flatten layer is (3136,).
    layers.Dense(128, activation="relu"),
    #The softmax function takes a vector of arbitrary real-valued inputs and normalizes them into
    # a probability distribution over the possible classes
    layers.Dense(10, activation="softmax"),
])

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    x_train, y_train,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_data=(x_test, y_test),
)

# Evaluate the model
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Save the model
model.save("fashion_mnist_model.h5")
