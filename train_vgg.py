# python train_vgg.py -d "/home/lili/Downloads/keras-libras/libras" -m "/home/lili/Downloads/keras-libras/output/smallvggnet.model"  -l "/home/lili/Downloads/keras-libras/output/smallvggnet.pickle" -p "/home/lili/Downloads/keras-libras/output/smallvggnet_plot.png"
# python train_vgg.py -d "C:/Users/lisan/Desktop/keras-libras/keras-libras/libras" -m "C:/Users/lisan/Desktop/keras-libras/keras-libras/output/smallvggnet.model"  -l "C:/Users/lisan/Desktop/keras-libras/keras-libras/output/smallvggnet.pickle" -p "C:/Users/lisan/Desktop/keras-libras/keras-libras/output/smallvggnet_plot.png"
import os
import cv2
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.smallvggnet import SmallVGGNet
import matplotlib

matplotlib.use("Agg")

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="/home/lili/Downloads/keras-libas/libras")
ap.add_argument("-m", "--model", required=True,
                help="/home/lili/Downloads/keras-libras/output/smallvggnet.model")
ap.add_argument("-l", "--label-bin", required=True,
                help="/home/lili/Downloads/keras-libras/output/smallvggnet.pickle")
ap.add_argument("-p", "--plot", required=True,
                help="/home/lili/Downloads/keras-libras/output/smallvggnet_plot.png")
args = vars(ap.parse_args())

# -------- Carregando os Dados --------------
print("[INFO] loading images...")
data = []
labels = []

imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

for imagePath in imagePaths:
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)

    # obter as labels
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)
# print ('Label -> '+ label)

# normalizar os dados no range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# -------- Treino e test --------------
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# -------- Aumento de imagens --------------
# gera novas imagens, baseada na antiga para mudando zoom, rotacao, girando...
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# -------- Construindo SmallVGGNet --------------
model = SmallVGGNet.build(width=64, height=64, depth=3, classes=len(lb.classes_))

INIT_LR = 0.01
EPOCHS = 75
BS = 32

print("[INFO] training network...")
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# -------- Treinando o Modelo --------------
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS)

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
plt.title("Training Loss and Accuracy (SmallVGGNet)")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["plot"])

print("[INFO] serializing network and label binarizer...")
model.save(args["model"])
f = open(args["label_bin"], "wb")
f.write(pickle.dumps(lb))
f.close()
