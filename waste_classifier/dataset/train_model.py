import tensorflow as tf
from tensorflow import keras
from keras.applications import MobileNetV2
from keras.layers import Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from pathlib import Path
import numpy as np

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20           # head training (early stopping will cut short if needed)
FINE_TUNE_EPOCHS = 15 # fine-tune backbone tail
LEARNING_RATE = 0.001

# Paths
BASE_DIR = Path(__file__).resolve().parent
train_dir = str(BASE_DIR / "train")
val_dir   = str(BASE_DIR / "validation")
test_dir  = str(BASE_DIR / "test")

# Stronger augmentation: flips, rotation, zoom, contrast, brightness, translation
data_augmentation = keras.Sequential([
    keras.layers.RandomFlip("horizontal_and_vertical"),
    keras.layers.RandomRotation(0.15),
    keras.layers.RandomZoom(0.15),
    keras.layers.RandomContrast(0.2),
    keras.layers.RandomBrightness(0.2),
    keras.layers.RandomTranslation(0.1, 0.1),
])

# Load data
print("Loading training data...")
train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

print("Loading validation data...")
val_data = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

print("Loading test data...")
test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

# Class names (sorted alphabetically by keras, same order as model output)
class_names = train_data.class_names
print(f"Classes: {class_names}")

# Compute class weights to handle imbalance (e.g. paper >> glass)
print("Computing class weights...")
class_counts = {}
for cls in class_names:
    cls_dir = Path(train_dir) / cls
    class_counts[cls] = len(list(cls_dir.glob("*")))
    print(f"  {cls}: {class_counts[cls]} images")

total_samples = sum(class_counts.values())
n_classes = len(class_names)
class_weight = {
    i: total_samples / (n_classes * class_counts[cls])
    for i, cls in enumerate(class_names)
}
print(f"Class weights: {class_weight}")

# Augment only training samples
print("Augmenting training data...")
train_data = train_data.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

# Normalize to [0, 1] + pipeline
def normalize(ds):
    return ds.map(
        lambda x, y: (x / 255.0, y),
        num_parallel_calls=tf.data.AUTOTUNE
    )

print("Setting up data pipeline...")
train_data = normalize(train_data).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_data   = normalize(val_data).cache().prefetch(tf.data.AUTOTUNE)
test_data  = normalize(test_data).cache().prefetch(tf.data.AUTOTUNE)

# Build model: MobileNetV2 backbone + larger head with Dropout
print("Building model...")
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)  # larger head than before (was 128)
x = Dropout(0.3)(x)                   # regularise to prevent overfitting
predictions = Dense(n_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

print("Compiling model (head training)...")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-7),
]

print("Model summary:")
model.summary()

# Phase 1 – train the head only
print("\n=== Phase 1: Training head ===")
model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=val_data,
    callbacks=callbacks,
    class_weight=class_weight,
)

# Phase 2 – fine-tune the last 50 layers of the backbone
print("\n=== Phase 2: Fine-tuning backbone (last 50 layers) ===")
base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_data,
    callbacks=callbacks,
    class_weight=class_weight,
)

# Evaluate
print("\n=== Evaluation on test set ===")
test_loss, test_acc = model.evaluate(test_data)
print(f"Test accuracy: {test_acc:.4f}  |  Test loss: {test_loss:.4f}")

# Save SavedModel
print("\nExporting SavedModel...")
model.export("waste_classifier_savedmodel")

# Convert to TFLite (dynamic-range quantisation for smaller/faster Pi inference)
print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_saved_model("waste_classifier_savedmodel")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

tflite_path = BASE_DIR / "waste_classifier.tflite"
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"TFLite model saved to {tflite_path}")

# Also save the Keras .h5 model
h5_path = BASE_DIR / "waste_classifier_model.h5"
model.save(h5_path)
print(f"Keras model saved to {h5_path}")

# Copy TFLite into the Pi scripts folder so it's ready to deploy
scripts_tflite = BASE_DIR.parent / "scripts" / "waste_classifier.tflite"
import shutil
shutil.copy2(tflite_path, scripts_tflite)
print(f"Copied TFLite to {scripts_tflite}")

print("\nAll done.")
