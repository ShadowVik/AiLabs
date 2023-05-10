import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv1D, Dense, Flatten, MaxPooling1D
from tensorflow.keras.models import Sequential
import os

# Load the spoken_digit dataset
(ds_train, ds_test), ds_info = tfds.load(
    'spoken_digit',
    split=['train[:80%]', 'train[80%:]'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

# Preprocess the dataset
def preprocess(audio, label):
    audio = tf.cast(audio, tf.float32) / 2**15
    audio = tf.pad(audio, paddings=((0, 30000 - tf.shape(audio)[0]),))
    audio = tf.expand_dims(audio, axis=-1)
    label = tf.one_hot(label, depth=10)
    return audio, label

ds_train = ds_train.map(preprocess).batch(32).cache().prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.map(preprocess).batch(32).cache().prefetch(tf.data.AUTOTUNE)

# Create the CNN model
model = Sequential([
    Conv1D(16, 3, activation='relu', input_shape=(30000, 1)),
    MaxPooling1D(2),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=20
)

# Save the model
model.save('spoken_digit_recognition.h5')

print("Model saved successfully.")