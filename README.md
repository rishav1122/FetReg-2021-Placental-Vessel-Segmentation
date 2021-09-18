# FetReg-2021-Placental-Vessel-Segmentation
# **The Novel approach for Biomedical Image Segmentation**

#### **Rishab Khantwal**

#### _Student at Indian Institute of Technology Bombay in the department of Mechanical Engineering_

#### Team name  - **RishabIITB**

#### Do you agree to make your submission public as part of the challenge archive? Yes

# **Background**

## Motivation 

#### Machines can augment analysis performed by radiologists, significantly reducing the time required to run diagnostic tests. The first step in many analysis pipelines is segmentation, occurring at several levels (e.g., separating nuclei, cells, tissues). This task has been an active field of research in image processing over the last 30 years. Various methods have been proposed and analyzed depending on the microscopy images' modality, quality, and resolution to research.

#### We have used U-Net architecture with ResNet101 backbone; U-Net can be used for any reasonable image masking task. It gives high accuracy on the effective training, adequate dataset, and training time. This architecture is input image size agnostic since it does not contain fully connected layers. 

#### Our approach focuses on the general improvement of the segmentation rather than improving models. It also focuses on providing an excellent image to the model. The motive was not to increase data but to improve the data. It involves enhancing data by preprocessing, a combination of the loss function and handling class imbalance.

## Dataset[1]

#### Fetoscopy videos acquired from the three different fetal medicine centers are first decomposed into frames, and excess black background is cropped to obtain squared images capturing mainly the fetoscope field-of-view. From each video, a subset of 100-150 non-overlapping informative frames was selected and manually annotated. All pixels in each image are labeled with 
_background (0),_
_placental vessel (1), _
_ablation tool (2), _
_fetus (3)_

${image?fileName=summary%2Epng&align=Center&scale=80&responsive=true&altText=dataset%5Fsummary}
Figure 1: Summary of the dataset


# **Methods**


## Preprocessing

#### The images downloaded were of low contrast and some images had a limited range of intensities. We use the contrast limited AHE (CLAHE) technique to improve contrast. It accomplishes this by effectively spreading out the most frequent intensity values. It differs from adaptive histogram equalization in its contrast limiting. In the case of CLAHE, the contrast limiting procedure is applied to each neighborhood from which a transformation function is derived. It prevents the over-amplification of noise that adaptive histogram equalization can give rise to. All the images were resized to 480 x 480 and normalized before passing to the model. The masks were interpolated using  Inter_Nearest otherwise ground truth changes due to interpolation

${image?fileName=Screenshot 2021%2D09%2D18 at 11%2E09%2E12 AM%2Epng&align=Center&scale=100&responsive=true&altText=finalclahe}

## Augmentation

#### Each time in training, different images were created using original images by applying augmentation on them. The techniques used in image augmentation were (1) Randomrotate90 which rotates images randomly in multiples of 90 degrees, and (2) Horizontal flip, which flips the image.  Since the scans would lie in almost the same color range, techniques like PCA_whitening and HSV_shifting were not used.
${image?fileName=augmentation%2Epng&align=Center&scale=100&responsive=true&altText=augmentation}


## Model Architecture

#### The U-Net[2] architecture was used with ResNet-101 as the backbone. The pre-trained Imagenet weights were used to initialize the model. The Image is continuously contracted as passed through convolution layers. The size goes from 480 x 480 to 30 x30 by continuously contracting into half in each step and then upsampled back to 480 x 480.

${image?fileName=The%2Darchitecture%2Dof%2DUnet%2Epng&align=Center&scale=100&responsive=true&altText=unet}
#### We also tried using other architectures such as LinkNet[5] and PSPNet[4]. For PSPNet, the images were resized to 384 x 384. The PSPNet got trained 5 times faster than LinkNet and U-Net. The PSPNet achieved the lowest mean IOU among all the architecture

${image?fileName=Screenshot 2021%2D09%2D18 at 12%2E25%2E17 PM%2Epng&align=Center&scale=100&responsive=true&altText=models}

${image?fileName=Screenshot 2021%2D09%2D18 at 12%2E13%2E59 PM%2Epng&align=Center&scale=100&responsive=true&altText=compare}
#### The Important point to note is that among the four classes U-Net with Resnet101 backbone has higher  IOU for only class 1, but it also achieves the highest MeanIOU because of consistent IOU over classes. U-Net with ResNet34 achieves the highest IOU over classes 0 and 3, but it has a significantly lower IOU on other two classes, which leads to the lower MeanIOU. 

##  Loss Function 

#### The dice loss was used along with Jaccard loss. Losses were scaled differently for each class because of class imbalance.

#### _Dice loss:_

#### The Dice score coefficient (DSC)[3] is a measure of overlap widely used to assess segmentation performance when a gold standard or ground truth is available. The 2-class variant of the Dice loss, denoted 
${image?fileName=dice%2Epng&align=Left&scale=100&responsive=true&altText=dice}


#### _Jaccard loss:_
#### The Intersection of Union for 2 class:


${image?fileName=jac%2Epng&align=None&scale=100&responsive=true&altText=jac}
####  Where yi and rn ∈ {0,1} is ground truth and  y^i and rn is the result of the model ,


${image?fileName=tot%2Epng&align=None&scale=100&responsive=true&altText=jac}
## Metrics and Training Parameters

#### Mean IOU (Intersection over union) and F1-score (Dice score) was used to evaluate the model. A threshold of 0.5 was used for both metrics.
${image?fileName=Screenshot 2021%2D09%2D18 at 11%2E23%2E22 AM%2Epng&align=Center&scale=100&responsive=true&altText=iou}


#### The model was trained on 85% of the dataset with 15% as validation. The objective was to maximize the validation IOU score. The learning scheduler was used to decrease the learning rate with the number of epochs with certain factors. The value of this factor was taken as 0.9. The model was trained on a computer with 3 NVIDIA GeForce GTX 1080 Ti with 11176MiB RAM each and 24 Intel(R) Xeon(R) CPU E5-2620 0 @ 2.00GHz processor. The model was trained on all 3 GPUs in mirror training mode. The model was trained for 50 epochs with a batch size of 8.
${image?fileName=withoutaugwithoutbce%5Floss2%2Epng&align=Center&scale=100&responsive=true&altText=loss}
## Testing random testing images
${image?fileName=Screenshot 2021%2D09%2D18 at 12%2E53%2E11 PM%2Epng&align=Center&scale=100&responsive=true&altText=test1}
${image?fileName=Screenshot 2021%2D09%2D18 at 12%2E55%2E16 PM%2Epng&align=Center&scale=80&responsive=true&altText=test2}

# **Conclusion/Discussion**

#### Our approach tries to explore different architectures and parameters for model training. The process also uses preprocessing, which improved the MeanIOU. The method also tests different loss functions and backbone for U-Net. The Mean-IOU is good, but IOU on each class is sparse and low for classes with fewer occurrences. Although our loss functions try to counter the imbalance to some extent, there is significant scope of improvements towards handling class imbalance. The different models achieving the highest IOU in other classes point that the ensembling of models could perform better and consistently over classes.


# **References**

[1] FETREG: PLACENTAL VESSEL SEGMENTATION AND REGISTRATION IN FETOSCOPY CHALLENGE DATASET [https://arxiv.org/abs/2106.05923](https://arxiv.org/abs/2106.05923)

[2] U-Net: Convolutional Networks for Biomedical Image Segmentation [https://arxiv.org/pdf/1505.04597v1.pdf](https://arxiv.org/pdf/1505.04597v1.pdf)

[3] Generalized Dice overlap as a deep learning loss function for highly unbalanced segmentations [https://arxiv.org/pdf/1707.03237.pdf](https://arxiv.org/pdf/1707.03237.pdf)

[4] Pyramid Scene Parsing Network [https://arxiv.org/abs/1612.01105v2](https://arxiv.org/abs/1612.01105v2)

[5] LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation [https://arxiv.org/pdf/1707.03718v1.pdf](https://arxiv.org/pdf/1707.03718v1.pdf)

[6] Deep Residual Learning for Image Recognition  [https://arxiv.org/abs/1512.03385v1](https://arxiv.org/abs/1512.03385v1)

[7] Segmentation models library [https://github.com/qubvel/segmentation_models](https://github.com/qubvel/segmentation_models)

​​@misc{Yakubovskiy:2019,

  Author = {Pavel Yakubovskiy},

  Title = {Segmentation Models},

  Year = {2019},

  Publisher = {GitHub},

  Journal = {GitHub repository},

  Howpublished = {\url{https://github.com/qubvel/segmentation_models}}

}
