#importing useful package
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

#the directory which datasets image are being kept
directories = r"C:\Users\bened\.keras\datasets\cats_and_dogs_filtered\train"
categories = ['screwdriver', 'spanner']

#read and join path for each category of screwdriver & spanner
for category in categories:
    path = os.path.join(directories, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break
print(img_array.shape)

#new size for the image after resizing
image_size = 200

#resize all the image using image_size variable provided above
new_array = cv2.resize(img_array, (image_size, image_size))
plt.imshow(new_array, cmap='gray')
plt.show()

#defining the training data
training_data = []

def create_training_data():
    for category in categories:
        path = os.path.join(directories, category)
        class_num = categories.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (image_size, image_size))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()

print(len(training_data))

#randomize the image shown
import random

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

x = []
y = []

for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1, image_size, image_size, 1)

#save the training data to be loaded on the training model
import pickle

pickle_out = open("x.pickle","wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()