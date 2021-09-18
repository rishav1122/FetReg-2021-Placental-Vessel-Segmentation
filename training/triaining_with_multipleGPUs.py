
#preprocessing


import tensorflow as tf
import segmentation_models as sm
import glob
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow  import keras 
sm.set_framework('tf.keras')

sm.framework()
# from keras.utils import normalize
from tensorflow.keras.metrics import MeanIoU
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

#Resizing images, if needed
SIZE_X = 480
SIZE_Y = 480
n_classes=4 #Number of classes for segmentation

#Capture training image info as a list
train_images = []
train_masks = []
for directory_path in glob.glob("Input/histimages/"):
    for img_path in glob.glob(os.path.join(directory_path, "*.png")):
        img = cv2.imread(img_path)       
        img = cv2.resize(img, (SIZE_Y, SIZE_X))
        train_images.append(img/255)
        tot = len(img_path)
        mask_path = "Input/labels/"+img_path.split("/")[-1]
        mask = cv2.imread(mask_path, 0)       
        mask = cv2.resize(mask, (SIZE_Y, SIZE_X), interpolation = cv2.INTER_NEAREST)  #Otherwise ground truth changes due to interpolation
        train_masks.append(mask)
print(len(train_images))  
#Convert list to array for machine learning processing        
train_images = np.array(train_images)

#Capture mask/label info as a list
 
        
print(len(train_masks))  
    
#Convert list to array for machine learning processing          
train_masks = np.array(train_masks)



np.unique(train_masks)

#################################################
#train_images = np.expand_dims(train_images, axis=3)
#train_images = normalize(train_images, axis=1)

train_masks_input = np.expand_dims(train_masks, axis=3)

#Create a subset of data for quick testing
#Picking 10% for testing and remaining for training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_images, train_masks_input, test_size = 0.15, random_state = 0)

#Further split training data t a smaller subset for quick testing of models
# X_train, X_do_not_use, y_train, y_do_not_use = train_test_split(X1, y1, test_size = 0.2, random_state = 0)
len(X_train)
print("Class values in the dataset are ... ", np.unique(y_train))  # 0 is the background/few unlabeled 

from tensorflow.keras.utils import to_categorical
train_masks_cat = to_categorical(y_train, num_classes=n_classes)
y_train_cat = train_masks_cat.reshape((y_train.shape[0], y_train.shape[1], y_train.shape[2], n_classes))



test_masks_cat = to_categorical(y_test, num_classes=n_classes)
y_test_cat = test_masks_cat.reshape((y_test.shape[0], y_test.shape[1], y_test.shape[2], n_classes))

del train_images 
del train_masks
del train_masks_input
# del  y_train
# del y_test
del train_masks_cat
del test_masks_cat


def scheduler(epoch, lr):
   if epoch < 40:
     return lr
   else:
     return lr * tf.math.exp(-0.1)

# Model building

# ######################################################
#Reused parameters in all models
with strategy.scope():

  n_classes=4
  activation='softmax'

  LR = 0.0001
  optim = tf.keras.optimizers.Adam(LR)
  jaccard_loss =  sm.losses.JaccardLoss(class_weights=[0.002, 1.5, 1.5, 1.5])
  total_loss =   jaccard_loss

  # actulally total_loss can be imported directly from library, above example just show you how to manipulate with losses
  # total_loss = sm.losses.binary_focal_dice_loss # or sm.losses.categorical_focal_dice_loss 

  metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]


  ########################################################################
  ###Model 1
  BACKBONE1 = 'resnet101'
  preprocess_input1 = sm.get_preprocessing(BACKBONE1)
  

  # preprocess input
  X_train1 = preprocess_input1(X_train)
  # print(len(X_train1)_
  X_test1 = preprocess_input1(X_test)
  # print(len(X_test1)_

  # define model
  model1 = sm.Unet(BACKBONE1, classes=n_classes, activation=activation)

  # compile keras model with defined optimozer, loss and metrics
  model1.compile(optim, total_loss, metrics=metrics)

  #model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=metrics)

  print(model1.summary())

train_data = tf.data.Dataset.from_tensor_slices((X_train1,y_train_cat)) 
val_data = tf.data.Dataset.from_tensor_slices((X_test1,y_test_cat)) 

batch_size = 8
train_data = train_data.batch(batch_size)
val_data = val_data.batch(batch_size)

options = tf.data.Options()
options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
train_data = train_data.with_options(options)
val_data = val_data.with_options(options)

### model training
path_checkpoint = 'weights/withoutaugmentationandbce_lr_norm2.hdf5'
modelckpt_callback = keras.callbacks.ModelCheckpoint(
    monitor="val_iou_score",
    filepath=path_checkpoint,
    verbose=1,
    mode = 'max',
    save_best_only=True,
)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

history1=model1.fit(train_data,
          epochs=20,
          verbose=1,
          validation_data=val_data,callbacks = [modelckpt_callback,lr_callback])




model1.save('without_augmentation_without_bce_lr_norm2.hdf5')


loss = history1.history['loss']
val_loss = history1.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig("withoutaugwithoutbce_loss2.png")
acc = history1.history['iou_score']
val_acc = history1.history['val_iou_score']
plt.show()

plt.plot(epochs, acc, 'y', label='Training IOU')
plt.plot(epochs, val_acc, 'r', label='Validation IOU')
plt.title('Training and validation IOU')
plt.xlabel('Epochs')
plt.ylabel('IOU')
plt.legend()
plt.savefig("withoutaugwithoutbce_iou2.png")
plt.show()

y_pred1=model1.predict(X_test1)
y_pred1_argmax=np.argmax(y_pred1, axis=3)


#Using built in keras function
#from keras.metrics import MeanIoUh
n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(y_test[:,:,:,0], y_pred1_argmax)
print("Mean IoU =", IOU_keras.result().numpy())


#To calculate I0U for each class...
values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
print(values)
class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

print("IoU for class1 is: ", class1_IoU)
print("IoU for class2 is: ", class2_IoU)
print("IoU for class3 is: ", class3_IoU)
print("IoU for class4 is: ", class4_IoU)


# y_pred1=model1.predict(X_train1)
# y_pred1_argmax=np.argmax(y_pred1, axis=3)


#Using built in keras function
#from keras.metrics import MeanIoUh
# n_classes = 4
# IOU_keras = MeanIoU(num_classes=n_classes)  
# IOU_keras.update_state(y_train[:,:,:,0], y_pred1_argmax)
# print("Mean IoU =", IOU_keras.result().numpy())
# 
# 
# #To calculate I0U for each class...
# values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)
# print(values)
# class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
# class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
# class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
# class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])
# 
# print("IoU for class1 is: ", class1_IoU)
# print("IoU for class2 is: ", class2_IoU)
# print("IoU for class3 is: ", class3_IoU)
# print("IoU for class4 is: ", class4_IoU)

#Test some random test images
import random
test_img_number = random.randint(0, len(X_test1)-1)
test_img = X_test1[test_img_number]
ground_truth=y_test[test_img_number]
test_img_input=np.expand_dims(test_img, 0)

test_img_input1 = preprocess_input1(test_img_input)

test_pred1 = model1.predict(test_img_input1)
test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,0], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(ground_truth[:,:,0], cmap='gray')
plt.subplot(233)
plt.title('Prediction on test image')
plt.imshow(test_prediction1, cmap='gray')
plt.savefig("withoutaugwithoutbce.png")