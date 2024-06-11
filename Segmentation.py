import tensorflow as tf # type: ignore
import tensorflow_datasets as tfds # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Load the dataset
dataset, info = tfds.load('cityscapes', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['validation']

# Preprocessing function
def preprocess(image, label):
    image = tf.image.resize(image, (513, 513)) / 255.0
    label = tf.image.resize(label, (513, 513), method='nearest')
    return image, label

# Apply preprocessing
train_dataset = train_dataset.map(preprocess).batch(8)
test_dataset = test_dataset.map(preprocess).batch(8)

# Load DeepLabV3+ model
base_model = tf.keras.applications.DenseNet121(
    input_shape=(513, 513, 3),
    include_top=False,
    weights='imagenet'
)

# Create the DeepLabV3+ model with custom classification head
def create_deeplabv3plus_model(num_classes):
    inputs = tf.keras.Input(shape=(513, 513, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.Conv2D(num_classes, (1, 1), activation='softmax')(x)
    x = tf.keras.layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(x)
    return tf.keras.Model(inputs, x)

model = create_deeplabv3plus_model(info.features['label'].num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=test_dataset)

# Evaluate the model
model.evaluate(test_dataset)

# Predict on test images
for images, labels in test_dataset.take(1):
    predictions = model.predict(images)
    # Convert predictions to class labels
    predicted_labels = tf.argmax(predictions, axis=-1)

    # Display some results
    for i in range(3):
        plt.subplot(3, 2, 2*i+1)
        plt.imshow(images[i])
        plt.title("Input Image")
        plt.subplot(3, 2, 2*i+2)
        plt.imshow(predicted_labels[i])
        plt.title("Predicted Segmentation")
    plt.show()