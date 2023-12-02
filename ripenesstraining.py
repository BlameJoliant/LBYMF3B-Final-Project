import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
import numpy as np

# Load the dataset from directories
img_width, img_height = 150, 150
train_data_dir = 'D:/Desktop/Academics/LBYMF3B/data/train/'
validation_data_dir = 'D:/Desktop/Academics/LBYMF3B/data/valid/'
nb_train_samples = 108
nb_validation_samples = 66
epochs = 30
batch_size = 10

# Set the input shape based on the image data format
if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
    print('Using channels_first')
else:
    input_shape = (img_width, img_height, 3)

# Define a learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 5 == 0 and epoch > 0:
        lr = lr / 2
    return lr

# Define the data generators with data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.4,
    height_shift_range=0.4,
    shear_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Use 'sparse' class mode for sparse_categorical_crossentropy
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse'
)

# Compute class weights manually
train_labels = train_generator.classes
class_counts = np.bincount(train_labels)
total_samples = np.sum(class_counts)
class_weights = {class_idx: total_samples / (len(class_counts) * class_counts[class_idx]) for class_idx in range(len(class_counts))}

# Define the validation data generator
validation_datagen = ImageDataGenerator(rescale=1./255)  # rescale the pixel values

# Use 'sparse' class mode for sparse_categorical_crossentropy
validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='sparse'
)

# Define the model architecture with increased complexity
model = Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

# Compile the model with an adjusted learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model checkpoint callback
model_checkpoint = ModelCheckpoint('model.h5', monitor='val_accuracy', save_best_only=True)

# Train the model and save the history
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[LearningRateScheduler(lr_scheduler), early_stopping, model_checkpoint],
    class_weight=class_weights,
    verbose=1
)

# Load the best model
best_model = tf.keras.models.load_model('model.h5')

# Save the trained model and weights
best_model.save('model.keras')
best_model.save_weights('weights.keras', overwrite=True)

# Evaluate the model on the validation set
test_loss, test_acc = best_model.evaluate(validation_generator, verbose=2)
print('Best Model Test accuracy:', test_acc)
