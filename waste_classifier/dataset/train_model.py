import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from pathlib import Path

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
FINE_TUNE_EPOCHS = 5
LEARNING_RATE = 0.001

# Paths
BASE_DIR = Path(__file__).resolve().parent
train_dir = str(BASE_DIR / "train")
val_dir = str(BASE_DIR / "validation")
test_dir = str(BASE_DIR / "test")

data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(0.05),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1)
])

# Load data
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

print("Running validating data...")
val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

print("Running test data...")
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

print("Augmenting training data...")
# Augment only training samples
train_data = train_data.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)


# Normalize + pipeline
print("Normalizing data and setting up pipeline...")
def normalize(ds):
    return ds.map(
        lambda x, y: (x / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

print("Setting up data pipeline...")
train_data = normalize(train_data).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_data = normalize(val_data).cache().prefetch(tf.data.AUTOTUNE)
test_data = normalize(test_data).cache().prefetch(tf.data.AUTOTUNE)

# Build model
print("Building model...")
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("Compiling model...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# Callbacks
print("Setting up callbacks to monitor training...")
callbacks = [
    keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
]

print("Model summary:")
model.summary()

# Train head
model.fit(train_data, epochs=EPOCHS, validation_data=val_data, callbacks=callbacks)

# Fine-tune backbone tail
print("Fine-tuning model...")
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_data, epochs=FINE_TUNE_EPOCHS, validation_data=val_data, callbacks=callbacks)

# Evaluate
print("Evaluating model on test data...")
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc:.4f}")

# Save (export a real SavedModel folder)
print("Exporting model...")
model.export("waste_classifier_savedmodel")

# Convert to TFLite
print("Converting model to TFLite format...")
converter = tf.lite.TFLiteConverter.from_saved_model("waste_classifier_savedmodel")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open("waste_classifier.tflite", "wb") as f:
    f.write(tflite_model)
print("Model converted to TFLite format.")