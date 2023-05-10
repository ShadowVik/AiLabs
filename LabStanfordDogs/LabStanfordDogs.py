import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models

# Load the Stanford Dogs dataset
(ds_train, ds_test), ds_info = tfds.load('stanford_dogs', split=['train', 'test'], shuffle_files=True, as_supervised=True, with_info=True)

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (224, 224))
    image = tf.cast(image, tf.float32)
    image /= 255.0
    return image, label

# Preprocess the data
ds_train = ds_train.map(preprocess).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

num_classes = ds_info.features['label'].num_classes

# Define the neural network architecture
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

# Train the model
def train_model(model, ds_train, ds_test, epochs=20):
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    history = model.fit(ds_train, epochs=epochs, validation_data=ds_test)

    return model, history

# Create the model
model = create_model()

# Train the model
trained_model, history = train_model(model, ds_train, ds_test)

# Save the trained model
trained_model.save('dogs_classification_model.h5')
