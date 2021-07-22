## GET AND SPLIT QMNIST DATA

To get the data from the model and make the split, you need to execute
```bash
python data/split_qmnist_ppml.py
```
This will produce a .pickle file inside the data folder with the image data and targets split in defender and reserve.
To create the tabular data, you just have to run:
```bash
python data/feature_extraction_qmnist_ppml.py
```
This will produce another .pickle file inside the data folder with the same split of defender and reserve, but now with the images in tabular format (feature representation produced by the second last layer of VGG-19 pre-trained on Imagenet).

## DEFENDER MODEL TRAINERS
The different scripts to fit and save machine learning models are stored in defender_model_trainers directory. The trained models are stored in a directory called defender_trained_models. For example:
```bash
python defender_model_trainers/simple_fn_keras.py
```