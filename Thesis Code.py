import numpy as np
import cv2
import glob
import scipy.special
import pandas as pd 
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.metrics import SpecificityAtSensitivity, SensitivityAtSpecificity, FalseNegatives, FalsePositives, TrueNegatives, TruePositives
from keras.models import load_model
import keras
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, roc_curve, auc, classification_report
from keras_preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, Flatten, Dropout, Activation, MaxPooling2D
from keras import layers
from keras import optimizers
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.applications.mobilenet import MobileNet
from keras_preprocessing.image import ImageDataGenerator
from imgaug import augmenters as iaa


X_data_Neg = []
X_data_Pos = []
Y_data_Neg = []
Y_data_Pos = []
image_dir = 'Input1/'
path, dirs, files = next(os.walk(image_dir))

pdata = pd.read_csv('CUHK_Real_Data.csv')
X = pdata['id']
Y = pdata['label']

#Runs through all the images, divided by 2 since the blocks should deal with both superficial and deep capillary plexus
for i in range(int(len(files)/2)):

    #This block deals with images with no DR progression
    if Y[i] == 0 and "(1)" in X[i]:
        
        #image sup is the image containing (1), while image deep contains the (2)
        image_sup = cv2.imread(image_dir + X[i] + '.png')
        image_deep = cv2.imread(image_dir + X[i+1] + '.png')
        
        #Load both segmented images so an average can be calculated
        seg_1 = cv2.imread(image_dir + X[i] + '_ves_prob.png')
        seg_2 = cv2.imread(image_dir + X[i+1] + '_ves_prob.png')

        #Convert all images to grayscale, since they will be single channels in the final images
        image_sup = cv2.cvtColor(image_sup, cv2.COLOR_BGR2GRAY)
        image_deep = cv2.cvtColor(image_deep, cv2.COLOR_BGR2GRAY)
        seg_1 = cv2.cvtColor(seg_1, cv2.COLOR_BGR2GRAY)
        seg_2 = cv2.cvtColor(seg_2, cv2.COLOR_BGR2GRAY)

        #To concatenate, we need to expand their dimensions
        seg_1 = np.expand_dims(seg_1, axis=-1)
        seg_2 = np.expand_dims(seg_2, axis=-1)

        #Maximum intensity projection for the segmented images since they will represent a single channel
        seg_MIP = np.concatenate((seg_1, seg_2), axis=-1)
        seg_MIP= np.max(seg_MIP, axis=-1)

        #Combine Superficial, Deep and Segmented images into a single 3 channel image
        image = np.concatenate((np.expand_dims(image_sup, axis=-1), np.expand_dims(image_deep, axis=-1), np.expand_dims(seg_MIP, axis=-1)), axis=-1)

        #Add images into a single 4D numpy array
        X_data_Neg.append(image)
        Y_data_Neg.append(Y[i])

    #This block deals with images with DR progression
    elif Y[i] == 1 and "(1)" in X[i]:
        #image sup is the image containing (1), while image deep contains the (2)
        image_sup = cv2.imread(image_dir + X[i] + '.png')
        image_deep = cv2.imread(image_dir + X[i+1] + '.png')
        
        #Load both segmented images so an average can be calculated
        seg_1 = cv2.imread(image_dir + X[i] + '_ves_prob.png')
        seg_2 = cv2.imread(image_dir + X[i+1] + '_ves_prob.png')

        #Convert all images to grayscale, since they will be single channels in the final images
        image_sup = cv2.cvtColor(image_sup, cv2.COLOR_BGR2GRAY)
        image_deep = cv2.cvtColor(image_deep, cv2.COLOR_BGR2GRAY)
        seg_1 = cv2.cvtColor(seg_1, cv2.COLOR_BGR2GRAY)
        seg_2 = cv2.cvtColor(seg_2, cv2.COLOR_BGR2GRAY)

        #To concatenate, we need to expand their dimensions
        seg_1 = np.expand_dims(seg_1, axis=-1)
        seg_2 = np.expand_dims(seg_2, axis=-1)

        #Maximum intensity projection for the segmented images since they will represent a single channel
        seg_MIP = np.concatenate((seg_1, seg_2), axis=-1)
        seg_MIP= np.max(seg_MIP, axis=-1)

        #Combine Superficial, Deep and Segmented images into a single 3 channel image
        image = np.concatenate((np.expand_dims(image_sup, axis=-1), np.expand_dims(image_deep, axis=-1), np.expand_dims(seg_MIP, axis=-1)), axis=-1)

        #Add images into a single 4D numpy array
        X_data_Pos.append(image)
        Y_data_Pos.append(Y[i])

X_data_Neg = np.asarray(X_data_Neg)
X_data_Pos = np.asarray(X_data_Pos)

# Decide in the train/validation split value
X_train_neg, X_val_neg, Y_train_neg, Y_val_neg =  train_test_split(X_data_Neg, np.zeros(X_data_Neg.shape[0]), test_size=0.2, random_state=42)
X_train_pos, X_val_pos, Y_train_pos, Y_val_pos =  train_test_split(X_data_Pos, np.ones(X_data_Pos.shape[0]), test_size=0.2, random_state=42)

X_val = np.concatenate((X_val_neg, X_val_pos), axis = 0)
Y_val = np.concatenate((Y_val_neg, Y_val_pos), axis = 0)
Y_val = np.expand_dims(Y_val, axis=-1)

### parameters for first image augmentation
seq = iaa.Sequential([
    iaa.Fliplr(0.5), 
    iaa.LinearContrast((0.75, 1.5)),
], random_order=True) 

### This for loop creates augmented data, and appends it into a single 4D numpy array

for i in range(int(X_train_neg.shape[0]/X_train_pos.shape[0])):
    images_aug = seq(images=X_train_pos)
    pos_aug_imgs = np.append(X_train_pos, images_aug, axis = 0)

#combining positive training data with augmented negative data so both classes have 410 images
X_train = np.concatenate((X_train_neg, pos_aug_imgs[0:X_train_neg.shape[0],:,:,:]), axis = 0)
print(X_train.shape)

for i in range(X_train.shape[0]):
    X_train[i,:,:] = (X_train[i,:,:] - np.amin(X_train[i,:,:]))/(np.amax(X_train[i,:,:]) - np.amin(X_train[i,:,:]))

### Initializing ground truth data
pos_GT = np.ones(X_train_pos.shape[0])
neg_GT = np.zeros(X_train_neg.shape[0])

Y_train = np.concatenate((neg_GT, pos_GT), axis=0)
Y_train = np.expand_dims(Y_train, axis=-1)
print('Y_train shape: ', Y_train.shape)
print('X_train shape: ', X_train.shape)


conv_base = ResNet50(
    include_top = True,
    weights='imagenet',
    input_shape = (224,224,3)
)
       
model = Sequential()
model.add(conv_base)
model.add(layers.Dense(256, use_bias=False))
model.add(layers.BatchNormalization())
model.add(layers.Activation("relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation = "sigmoid"))

conv_base.summary()
model.summary()

# # This is the block to "freeze" certain layers, but the final result was better with no layer freezing 
# conv_base.Trainable=True
# set_trainable=False
# for layer in conv_base.layers:
#     if layer.name == 'res5a_branch2a':
#         set_trainable = True
#     if set_trainable:
#         layer.trainable = True
#     else:
#         layer.trainable = False

model.compile(optimizers.Adam(1e-5), loss = "binary_crossentropy", metrics=["accuracy"])
earlystopper = EarlyStopping(monitor='val_loss', patience=15, verbose=1)

history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), batch_size=8, epochs=150, callbacks=[earlystopper])

############### Block to output Accuracy and Loss plots ################################
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig('Accuracy')
plt.close()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
plt.savefig('Loss')
#######################################################################################

# make prediction on the test data using the best model
predictions = model.predict(
    X_val,
    batch_size=1,
    verbose=1)

predicted_classes = predictions > 0.5

print('shape of:')
print(X_val.shape)

FN = FalseNegatives()
FN.update_state(Y_val, predicted_classes)
FN = FN.result().numpy()

FP = FalsePositives()
FP.update_state(Y_val, predicted_classes)
FP = FP.result().numpy()

TN = TrueNegatives()
TN.update_state(Y_val, predicted_classes)
TN = TN.result().numpy()

TP = TruePositives()
TP.update_state(Y_val, predicted_classes)
TP = TP.result().numpy()

# calculate specificity at sensitivity and sensitivity at specificity (hold at 90%)
specAtSens_val = SpecificityAtSensitivity(0.9)
specAtSens_val.update_state(Y_val, predicted_classes)
specAtSens_val = specAtSens_val.result().numpy()

sensAtSpec_val = SensitivityAtSpecificity(0.9)
sensAtSpec_val.update_state(Y_val, predicted_classes)
sensAtSpec_val = sensAtSpec_val.result().numpy()

classification_matrix = classification_report(Y_val, predicted_classes, target_names=['No Progression', 'Progression'])

# epsilon value to avoid division by zero for PPV and NPV (if it never guesses either positive or negative [i.e. greatly overfitting or unbalanced dataset])
epsilon = 0.000001

# calculate evaluation metrics (Accuracy, Sepcificity, Sensitivity, PPV, NPV)
Acc_t, Sens_t, Spec_t = (TN+TP)/(TN+TP+FN+FP), TP/(TP + FN), TN/(TN + FP)
PPV_t, NPV_t = round(TP/(TP + FP + epsilon),6), round(TN/(FN + TN + epsilon),6)

# calculate ROC values
fpr, tpr, _ = roc_curve(Y_val, predictions)
roc_auc = auc(fpr, tpr)

fold_str = 'Sens and Spec'
folder_path = 'Metrics.txt'

# print metrics
with open(folder_path,"a") as file1:
    file1.write('Classification metrics for ' + fold_str + ' : ' + '\n')
    file1.write('Accuracy: ' + str(Acc_t) + '\n') 
    file1.write('Sensitivity :' + str(Sens_t) + '\n') # TP/(TP+FN) or true positive over all positives
    file1.write('Specificity :' + str(Spec_t) + '\n') # TN/TN+FP or true negative over all negatives
    file1.write('PPV :' + str(PPV_t) + '\n') # TP/(TP+FN) or true positive over all positives
    file1.write('NPV :' + str(NPV_t) + '\n') # TN/TN+FP or true negative over all negatives
    file1.write('AUC of ROC :' + str(roc_auc) + '\n') # AUC of ROC
    file1.write('Specificity at sensitivity of 0.9 :' + str(specAtSens_val) + '\n') # specificity at sensitivity of 90%
    file1.write('Sensitivity at specificity of 0.9 :' + str(sensAtSpec_val) + '\n') # sensitivity at specificity of 90%
    file1.write('Classification matrix :' + classification_matrix + '\n') # sensitivity at specificity of 90%


