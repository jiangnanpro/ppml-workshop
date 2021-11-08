## GET AND SPLIT QMNIST DATA

To get the data from the model and make the split, you need to execute
```bash
python data/split_qmnist.py
```
This will produce a .pickle file inside the data folder with the image data and targets split in defender and reserve.

To create the tabular data using the second last layer of a VGG19 pretrained on Imagenet, you just have to run:
```bash
python data/feature_extraction1.py
```
To create the tabular data with a ResNet50 pretrained on Fake-MNIST ([credits to Haozhe Sun](https://github.com/SunHaozhe)), run instead:
```bash
python data/feature_extraction2.py
```
Any of these two scripts will produce another .pickle file inside the data folder with the same split of defender and reserve, but now with the images in tabular format.

## GET CIFAR10 DATA

Preprocessed tabular data for CIFAR10 can be downloaded from: https://drive.google.com/file/d/17bwMCqSt-6dTxft6lGtEIPQrS2dRrQ1R/view?usp=sharing

## BLACK-BOX ATTACKER DATA

The script generate_table_v1.py is used to generate the table for the black-box attacker with the different sklearn algorithms.

## DEFENDER MODEL TRAINERS
The different scripts to fit and save machine learning models are stored in defender_model_trainers directory. The trained models are stored in a directory called defender_trained_models. For example:
```bash
python defender_model_trainers/simple_fn_keras.py
```