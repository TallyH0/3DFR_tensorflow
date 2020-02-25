# 3DFR_tensorflow
Background subtraction using tensorflow

This is a tensorflow implementation of [3DFR: A Swift 3D Feature Reductionist Framework for Scene Independent Change Detection](https://arxiv.org/abs/1912.11891)


you can download CDNET2014 Datset [here](http://jacarini.dinf.usherbrooke.ca/dataset2014/)

## Requirement
tensorflow-gpu 1.13.2

opencv-python

## Result
baseline category : train [office, highway, PETS2006] , test [pedestrian]
<img src='figure/baseline.gif'></img>

## Train
```
python train.py --config <config file path>
```

config file is
```
[DEFAULT]
dir_data = [
  "<CDNET_DATASET_PATH>/dataset/cameraJitter/badminton",
  "<CDNET_DATASET_PATH>/dataset/cameraJitter/sidewalk",
  "<CDNET_DATASET_PATH>/dataset/cameraJitter/boulevard"
  ]
dir_model = model_CDNet2014_cameraJitter
img_h = 360
img_w = 240
gray = True
batch_size = 1
max_epoch = 100
```

put your <CDNET_DATASET_PATH> in config file

## Test
```
python test.py --config <config file path> --input_dir <CDNet category dir> --output_dir <output_dir path>
```
