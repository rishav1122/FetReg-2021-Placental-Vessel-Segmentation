"""
Fetoscopy placental vessel segmentation and registration challenge (FetReg)
EndoVis - FetReg2021 - MICCAI2021
Challenge link: https://www.synapse.org/#!Synapse:syn25313156

Task 1 - Segmentation - Docker dummy example showing 
the input and output folders for the submission
"""

import sys  # For reading command line arguments
from glob import glob  # For listing files
import cv2  # For handling image loading & processing
import os  # for path creation
import numpy as np
import keras
import segmentation_models as sm
sm.set_framework('tf.keras')
import matplotlib.pyplot as plt
sm.framework()
INPUT_PATH =  sys.argv[1]
OUTPUT_PATH = sys.argv[2]
print("yess")
BACKBONE1 = 'resnet34'
preprocess_input1 = sm.get_preprocessing(BACKBONE1)
n_classes = 4
activation='softmax'
from tensorflow.keras import layers
# import keras.Functional
# model1 = sm.Unet(BACKBONE1, classes=n_classes, activation=activation)
model1 = keras.models.load_model("code/final.hdf5",custom_objects={'jaccard_loss': sm.losses.jaccard_loss,'dice_loss': sm.losses.dice_loss,'dice_loss_plus_2focal_loss_plus_2jaccard_loss':sm.losses.dice_loss+2*sm.losses.categorical_focal_loss+2*sm.losses.jaccard_loss,'iou_score' : sm.metrics.iou_score,'f1-score':sm.metrics.f1_score})


if __name__ == "__main__":
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
        print(OUTPUT_PATH +' created')
    else:
        print(OUTPUT_PATH +' exists')

        
    input_file_list = glob(INPUT_PATH + "/*.png")
    print("yo")

    for f in input_file_list:
        file_name = f.split("/")[-1]
        img = cv2.imread(f)  
        test_img = cv2.resize(img, (480, 480))
        print("yess")
        test_img_input=np.expand_dims(test_img, 0)
        test_img_input1 = preprocess_input1(test_img_input)
        test_pred1 = model1.predict(test_img_input1)
        test_prediction1 = np.argmax(test_pred1, axis=3)[0,:,:]
     	
      	
        
#         ret,img2 = cv2.threshold(img,100,1,cv2.THRESH_BINARY)
        
        
        
#         img2 = np.uint8(img2)
        result = cv2.imwrite(OUTPUT_PATH + "/" + file_name, test_prediction1)
#         plt.imshow(test_prediction1, cmap='gray')
#         plt.savefig(f)

        if result==True:
            print(OUTPUT_PATH+'/' +file_name +' output mask saved')
        else:
            print('Error in saving file')
