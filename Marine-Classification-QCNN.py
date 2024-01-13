import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def load_images_from_folder(root_folder, image_size=(128, 128)):
    images = []
    labels = []
    label_dict = {}

    for label, folder_name in enumerate(os.listdir(root_folder)):
        label_dict[label] = folder_name
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, image_size)
                    images.append(img)
                    labels.append(label)

    return images, labels, label_dict

data_folder = r'C:\Users\thesu\Downloads\archive'

images, labels, label_dict = load_images_from_folder(data_folder)

images = np.array(images)
labels = np.array(labels)

train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

train_images = train_images / 255.0
test_images = test_images / 255.0

print("Numerical Labels and Corresponding Classes:")
for label, class_name in label_dict.items():
    print(f"Label {label}: {class_name}")

print("Shape of training images:", train_images.shape)
print("Shape of testing images:", test_images.shape)
print("Shape of training labels:", train_labels.shape)
print("Shape of testing labels:", test_labels.shape)

train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
