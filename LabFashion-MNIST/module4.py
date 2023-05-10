import tensorflow as tf
import tensorflow_datasets as tfds

# TPU initialization code
resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
print("All devices: ", tf.config.list_logical_devices('TPU'))

# Define the Keras model
def create_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(256, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(256, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
    ])

# Load the dataset
def get_dataset(batch_size, is_training=True):
    split = 'train' if is_training else 'test'
    dataset, info = tfds.load(name='mnist', split=split, with_info=True,
                              as_supervised=True, try_gcs=True)

    def scale(image, label):
        image = tf.cast(image, tf.float32)
        image /= 255.0
        return image, label

    dataset = dataset.map(scale)

    if is_training:
        dataset = dataset.shuffle(10000)
        dataset = dataset.repeat()

    dataset = dataset.batch(batch_size)

    return dataset

# Training the model
batch_size = 64
steps_per_epoch = 60000 // batch_size
validation_steps = 10000 // batch_size

strategy = tf.distribute.TPUStrategy(resolver)

with strategy.scope():
    model = create_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['sparse_categorical_accuracy'])

train_dataset = get_dataset(batch_size)
test_dataset = get_dataset(batch_size, is_training=False)

model.fit(train_dataset,
          epochs=10,
          steps_per_epoch=steps_per_epoch,
          validation_data=test_dataset,
          validation_steps=validation_steps)

model.save("mnist_model_tpu.h5")
