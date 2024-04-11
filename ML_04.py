import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
BATCH_SIZE = 32
NUM_CLASSES = 3  
EPOCHS = 10  

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = datagen.flow_from_directory(
    'ML_04/train',  
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

validation_generator = datagen.flow_from_directory(
    'ML_04/validation',  
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical')

test_generator = datagen.flow_from_directory(
    'ML_04/test',  
    target_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False)

steps_per_epoch_train = len(train_generator)
steps_per_epoch_validation = len(validation_generator)
steps_per_epoch_test = len(test_generator)

history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch_train,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=steps_per_epoch_validation)

test_loss, test_acc = model.evaluate(test_generator, verbose=0)
print('Test accuracy:', test_acc)
test_results = model.predict(test_generator)
class_labels = train_generator.class_indices.keys()
print("Accuracy for each sign image:")

class_accuracies = []

for i, class_label in enumerate(class_labels):
    class_index = train_generator.class_indices[class_label]
    class_results = test_results[test_generator.classes == class_index]
    class_accuracy = sum(class_results.argmax(axis=1) == i) / len(class_results) * 100
    class_accuracies.append(class_accuracy)
    print(f"{class_label}: {class_accuracy:.2f}%")

# Plotting results
plt.figure(figsize=(10, 5))

# Bar chart
plt.subplot(1, 2, 1)
plt.bar(class_labels, class_accuracies, color='skyblue')
plt.xlabel('Food Items')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy for Each Food Item')

# Pie chart
plt.subplot(1, 2, 2)
plt.pie(class_accuracies, labels=class_labels, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral', 'lightskyblue'])
plt.title('Accuracy Distribution')

plt.tight_layout()
plt.show()
