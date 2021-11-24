
# AE-NE : Autoencoder with background noise estimation for background reconstruction and foreground segmentation 

Implementation of the model AE-NE described in the paper "
Autoencoder-based background reconstruction and foreground segmentation with background noise estimation"




https://user-images.githubusercontent.com/44267731/143236111-0354801d-f0e4-4500-89b7-c08ea39be24f.mp4



## Requirements

The model needs Pytorch (>= 1.7.1) and Torchvision with cuda capability (see https://pytorch.org/ )

The model also needs OpenCV (>=4.1) (see https://opencv.org/ )


To install other requirements:

```setup
pip install -r requirements.txt
```
The model has been tested on Nvidia RTX 2080 TI and Nvidia RTX 3090 GPU.

## How to use the model

the command to generate the backgrounds and foreground masks from a sequence of frames is 

```
python main.py --input_path your_input_path 
```

where your_input_path is the path to the folder where the frame sequence is saved.
Example : python main.py --input_path /workspace/Datasets/CDnet2014/dataset/baseline/highway

the result background images and foreground masks will be stored in two subdirectories 'results' and 'backgrounds' the current working directory.

To view options, type python main.py -h

The default training mode is fully unsupervised. A weakly supervised option is also implemented, where the number of training iterations and the background complexity have to be provided as inputs to the model.

## Evaluation

To evaluate the AE-NE model on the CDnet 2014 dataset: 

- download the CDnet 2014 dataset from the following link : 
```
http://jacarini.dinf.usherbrooke.ca/static/dataset/dataset2014.zip
```
and save it in some folder 

- update the dataset path in the end of the python file "test_CDnet.py"
- to perform a  partial test, update the category list in the end of the python file "test_CDnet.py"
- run the python program test_CDnet.py

Warning : Different runs of the model with the same inputs may lead to small differences in evaluation results compared to the published results  due to the random initialization of the autoencoder and the random sampling of the images during training.








