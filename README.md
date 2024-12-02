# Smart-Home-Platform-for-Stroke-Rehabilitation
The supporting code and data for "A Unified Platform for At-Home Post-Stroke Rehabilitation Enabled by Wearable Technologies and Artificial Intelligence"

This repository provides supporting data and code related to the above research, which will be made publicly available at the time of publication.

## Motor-rehabilitation-states-monitoring
The code is used to train and test the model for classifying motor impairment rehabilitation states. Due to data privacy concerns and the patients' own consent, only a subset of patients' plantar pressure data is provided. However, detailed descriptions of the device setup and experimental conditions required for full reproduction can be found in the manuscript and supplementary information. Just one sample was uploaded for demo, more samples will be available after publication.

## Action-classifier-model
1. System requirements
Win10 and above
Python3.10 and above
2. Installation guideInstructions
It takes about 1 hour to complete the environment configuration.
3. Demo
Finally output a model weight file pose-classification-dense-model.h5
4. instructions for use
You need to install related libraries in pycharm/VS code first, including Numpy and tensorflow, put X_train.npy, y_train.npy, and MLP_train.py into a folder, and run MLP_train.py to train the action classification model.
