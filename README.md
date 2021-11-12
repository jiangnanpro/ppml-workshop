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

## WHITE-BOX ATTACKER 

* All the defender models and input data needed to run the experiments are available at [https://upsud-my.sharepoint.com/:u:/g/personal/haozhe_sun_u-psud_fr/ERZ_x4Yj_IpAtR2ctr_7o_0BguzXXgZv2fkJwh4iGUW7BQ?e=Cb09dJ](https://upsud-my.sharepoint.com/:u:/g/personal/haozhe_sun_u-psud_fr/ERZ_x4Yj_IpAtR2ctr_7o_0BguzXXgZv2fkJwh4iGUW7BQ?e=Cb09dJ)
* `QMNIST_ppml.pickle` is also available by running `data/split_qmnist.py`
* `best_model_supervised_resnet50_QMNIST_defender_whole-0.0001-normal-normal_gallant-wildflower-1.pth` is also available at [https://upsud-my.sharepoint.com/:u:/g/personal/haozhe_sun_u-psud_fr/EQzSwHgzCQxIuKPkthZZX3YB2-RaYGpjAbiZW_ZBpqMCRA?e=Wzy9H7](https://upsud-my.sharepoint.com/:u:/g/personal/haozhe_sun_u-psud_fr/EQzSwHgzCQxIuKPkthZZX3YB2-RaYGpjAbiZW_ZBpqMCRA?e=Wzy9H7)
* Defender models were trained using scripts `train_supervised.sh` and `supervised_train_resnet50_defender.py`. 
* Once the defender models and input data are ready, `oracle_attack_UDA.sh` and `oracle_attack_UDA.py` allows generating the input features for the white-box attackers without using a neural network.
* `oracle_attack_using_NN.sh` and `oracle_attack_using_NN.py` allows generating the input features for the white-box attackers using a neural network.
* Once the previous step is done, the script `compute_results_hz.py` allows generating the final table for the white-box attack experiments, where the utility scores are computed by the script `get_reserve_accuracy_.py`



















