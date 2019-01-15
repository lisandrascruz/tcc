# run: python3 train_simple_nn.py -d "/home/lili/Downloads/keras-libras/libras" -m "/home/lili/Downloads/keras-libras/output/simple_nn.model" -l "/home/lili/Downloads/keras-libras/output/simple_nn_lb.pickle" -p "/home/lili/Downloads/keras-libras/output/simple_nn_plot.png"
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
ap.add_argument("-d", "--dataset", required=True,
                help="/home/lili/Downloads/keras-libras/libras")
ap.add_argument("-m", "--model", required=True,
                help="/home/lili/Downloads/keras-libras/output")
ap.add_argument("-l", "--label-bin", required=True,
                help="/home/lili/Downloads/keras-libras/output")
ap.add_argument("-p", "--plot", required=True,
                help="/home/lili/Downloads/keras-libras/output")
args = vars(ap.parse_args())

print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"]))) #???
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    i = i+1
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (32, 32)).flatten()
    data.append(image)

    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
    ''' print (str(i) + ' - img: ' + imagePath + ' label: ' + label) '''

# normaliza para o intervalo [0, 1] (Normalização)
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# ---------------------2.Treino e teste -------------------

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.3, random_state=42)

# converte os inteiros que representam as labels em vetores (???)
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
model.compile(loss="categorical_crossentropy", optimizer=opt,
	metrics=["accuracy"])
H = model.fit(trainX, trainY, validation_data=(testX, testY),
    epochs=EPOCHS, batch_size=32)

# evaluate the network
# precision: das clasificadas como x, qts realmente eram x
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
 
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

