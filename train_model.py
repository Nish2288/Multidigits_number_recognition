import cv2
import numpy as np
from captchahelper import preprocess
from imutils import paths
from keras.preprocessing.image import img_to_array
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from lenet import LeNet
from keras.preprocessing.image import ImageDataGenerator

#   from keras.optimizers import Adam


imgpaths = list(paths.list_images('training_data'))
#print(imgpaths)
data = []
labels = []

for imagePath in imgpaths:
    #print(imagePath)
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = preprocess(image, 28, 28)
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
	labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
lb = LabelBinarizer().fit(trainY)
trainY = lb.transform(trainY)
testY = lb.transform(testY)

# Data Augmentation
aug =  ImageDataGenerator(rotation_range=30, height_shift_range=0.2, width_shift_range=0.2, shear_range=0.2)

# initialize the model
print("[INFO] compiling model...")
model = LeNet.build(width=28, height=28, channels=1, classes=9)
#opt = SGD(lr=0.01)
model.compile(loss="categorical_crossentropy", optimizer='adam',
	metrics=["accuracy"])

# train the network
print("[INFO] training network...")
H=model.fit_generator(aug.flow(trainX, trainY, batch_size =32), validation_data=(testX, testY), steps_per_epoch = len(trainX)//32, epochs=10, verbose=1)


# save the model to disk
print("[INFO] serializing network...")
model.save('captcha.h5')