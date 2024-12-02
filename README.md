# Smart-Home-Platform-for-Stroke-Rehabilitation
The supporting code and data for "A Unified Platform for At-Home Post-Stroke Rehabilitation Enabled by Wearable Technologies and Artificial Intelligence"

This repository provides supporting data and code related to the above research, which will be made publicly available at the time of publication.

## Motor-rehabilitation-states-monitoring
The code is used to train and test the model for classifying motor impairment rehabilitation states. Due to data privacy concerns and the patients' own consent, only a subset of patients' plantar pressure data is provided. However, detailed descriptions of the device setup and experimental conditions required for full reproduction can be found in the manuscript and supplementary information.

1. Requirement: Python 3.10 or above
Equipped with pytorch and other necessary libraries

2. Demo: Finally output a class label (0, 1, 2)

3. Instructions for use: follow the notes in the code

4. Installation guide
It takes about 1 hour to complete the environment configuration.


## Action-classifier-model
1. System requirements
Win10 and above
Python3.10 and above
2. Installation guide
It takes about 1 hour to complete the environment configuration.
3. Demo
Finally output a model weight file pose-classification-dense-model.h5
4. instructions for use
You need to install related libraries in pycharm/VS code first, including Numpy and tensorflow, put X_train.npy, y_train.npy, and MLP_train.py into a folder, and run MLP_train.py to train the action classification model.

## Object-detection-model
1. System requirements
Win10 and above
Python3.10 and above
2. Installation guide
It takes about 1 hour to complete the environment configuration 
3. Demo
Finally output a model weight file best.pt
4. instructions for use
You need to download the yolov8 resource files from https://github.com/ultralytics, use the data.yaml and yolov8n.ymal files in the ultralytics-main folder, set the dataset address, and run the train.py file to start training the model.

Running time of the above modules are also within a 1s (except for training process).

Note: Only small datasets was uploaded for demo, more samples will be available after publication.
