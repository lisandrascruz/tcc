# python train_test_simple_nn.py -dTrain "A:/Projetos/TCC/tcc/libras-p1" -dTest "A:/Projetos/TCC/tcc/libras-p2" -m "A:/Projetos/TCC/tcc/output/smallvggnet.model"  -l "A:/Projetos/TCC/tcc/output/smallvggnet.pickle" -p "A:/Projetos/TCC/tcc/output/smallvggnet_plot.png"

import matplotlib
matplotlib.use("Agg") #salvar plotagens no disco
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import pickle
import cv2
import os

i = 0
# ---------------------1.Carregando Dados-------------------
ap = argparse.ArgumentParser()
ap.add_argument("-dTrain", "--datasetTrain", required=True,
                help="A:/Projetos/TCC/tcc/libras-p2")
ap.add_argument("-dTest", "--datasetTest", required=True,
                help="A:/Projetos/TCC/tcc/libras-p1")
ap.add_argument("-m", "--model", required=True,
                help="/home/lili/Downloads/keras-libras/output")
ap.add_argument("-l", "--label-bin", required=True,
                help="/home/lili/Downloads/keras-libras/output")
ap.add_argument("-p", "--plot", required=True,
                help="/home/lili/Downloads/keras-libras/output")
args = vars(ap.parse_args())

print("[INFO] loading images...")
dataTrain = []
dataTest = []
labelsTrain = []
labelsTest = []

imagePathsTrain = sorted(list(paths.list_images(args["datasetTrain"])))
imagePathsTest = sorted(list(paths.list_images(args["datasetTest"])))
random.seed(42)
random.shuffle(imagePathsTrain)
random.shuffle(imagePathsTest)

for imagePath in imagePathsTrain:
    i = i+1
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    dataTrain.append(image)

    labelTrain = imagePath.split(os.path.sep)[-2]
    labelsTrain.append(labelTrain)
print("QTDD TRAIN:" +str(len(labelsTrain)))

for imagePath in imagePathsTest:
    i = i+1
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    dataTest.append(image)

    labelTest = imagePath.split(os.path.sep)[-2]
    labelsTest.append(labelTest)
print("QTDD TEST:" +str(len(labelsTest)))

# normaliza para o intervalo [0, 1] (Normalização)
dataTrain = np.array(dataTrain, dtype="float") / 255.0
dataTest = np.array(dataTest, dtype="float") / 255.0
labelsTrain = np.array(labelsTrain)
labelsTest = np.array(labelsTest)

# ---------------------2.Treino e teste -------------------
trainX = dataTrain
testX = dataTest
trainY = labelsTrain
testY = labelsTest

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# ---------------------3.Modelo-------------------
# define a arquitetura usada Keras 
model = Sequential()
model.add(Dense(1024, input_shape=(3072,), activation="sigmoid"))
model.add(Dense(512, activation="sigmoid"))
model.add(Dense(len(lb.classes_), activation="softmax"))

# ---------------------4.Compilando o modelo Keras-------------------
INIT_LR = 0.01 # initial learning rate 
EPOCHS = 75

print("[INFO] training network...")
opt = SGD(lr=INIT_LR)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY), epochs=EPOCHS, batch_size=32)

# evaluate the network
# precision: das clasificadas como x, qts realmente eram x
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))
 
# plot the training loss and accuracy
N = np.arange(0, EPOCHS)
plt.style.use("ggplot")
plt.figure()
plt.plot(N, H.history["loss"], label="train_loss")
plt.plot(N, H.history["val_loss"], label="val_loss")
plt.plot(N, H.history["acc"], label="train_acc")
plt.plot(N, H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy (Simple NN)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()

