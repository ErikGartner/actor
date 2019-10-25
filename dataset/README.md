# Dataset

This readme contains the instructions for downloading, extracting and cleaning the dataset used in the ACTOR paper. It also provides the steps to compute pose predictions, deep features and instance features required for the ACTOR agent. All pose predictions are pre-computed to speed-up RL training.

Main requirements are Python 3.6 and a CUDA-enabled GPU for computing OpenPose features using Tensorflow (see details below).

## Dataset structure

ACTOR assumes that the dataset and the associated cache of pose predictions are provided using a certain structure.

There are two important folders:
- The _data_ folder that contains the raw images and Panoptic annotations (used for evaluating the 3d error).
- The _cache_ folder that contains the OpenPose predictions, deep features and instance features used for matching.

Internally these two folders mirror each other. They assume a second level of folders that separate the scenes into train, test and validation splits.

Example:
```
data/
 train/
   scene1/
 val/
   scene2/
 test/
   scene3/

cache/
 train/
   scene1/
 val/
   scene2/
 test/
   scene3/
```

## Creating the dataset

#### 0. Create folder structure

You'll need around 500 - 800 gb of free space on your target drive. You may split the `data/` and `cache/` folder into different drives.

1. Create your data folder (e.g. `data/`) somewhere. Do _not_ create any subfolders.
2. Create your cache folder (e.g. `cache/`) somewhere. Then create `cache/test`, `cache/train`, `cache/val`.
3. Go into the `load_config.m` file and set the `dataset_path` and `dataset_cache` flags to the corresponding paths (they should point to `data/` and `cache/`, respectively).

#### 1. Download and clean Panoptic data

1. Download the scenes and extract annotations and images:
    - `./downloadAll.sh <data-folder-path>`
2. Clean-up the scene to remove bad camera frames and ensure that each scene has a fixed number of persons:
    - `python3 clean_up_panoptic.py --check-same-nbr-frames  --check-green-frame --hd --same-people --split-scene --min-con 100 --delete --min-nbr-people 1 <data-folder-path>`
3. Split the data into the official train, validation and test split by running:
    - `bash split_data.sh <data-folder-path>`.

#### 2. Compute OpenPose predictions and deep features

To predict the 2d joints and deep features we use a Tensorflow port of OpenPose based off [this example](https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-1-7dd4ca5c8027). Your computer will need to support CUDA 9.0 or you'll need to install the `tensorflow` instead of `tensorflow-gpu` Python package to compute all features on the CPU.

1. Go to `openpose-tf/`. In the following steps all paths are relative to that folder.
2. Download [weights](https://lu.box.com/s/1pugs9ln2q5brd5043l9risx9hayqpg7) to `models/`.
3. Install Python 3.6 and the requirements in `requirements.txt` (or use [Pipenv](https://github.com/pypa/pipenv) to do this automatically).
4. For each split (train, val, test) compute the 2d pose predictions and deep features.
    - `python3 cache_pano.py --panopticpath <data-folder-path>/train --cachepath <cache-folder-path>/train`
    - `python3 cache_pano.py --panopticpath <data-folder-path>/val --cachepath <cache-folder-path>/val`
    - `python3 cache_pano.py --panopticpath <data-folder-path>/test --cachepath <cache-folder-path>/test`
5. Merge and bilinearly resize the deep features for each cache split using Matlab script `resize_merge.m`:
    - `resize_merge('<cache-folder-path>/train')`
    - `resize_merge('<cache-folder-path>/val')`
    - `resize_merge('<cache-folder-path>/test')`

#### 3. Compute instance features for matching

Next we want to generate the instance features used for matching people in the scene by appearance. The model is first trained for 40k iterations on the training split, then fine-tuned 2k iterations for each individual scene.

The base of the instance features comes from a VGG-19 model.

1. Download the VGG-19 [weights](https://lu.box.com/s/eswxcfj9gn2qhjp8o7o6mxrihcgrvkpx) (sha1: 7e1441c412647bebdf7ae9750c0c9aba131a1601).
2. Either run the 40k base training on the _train_ split or download the weights.
    - Train weights from scratch using the Matlab script: `run_train_instance_detector('train')`
    - Download pre-trained 40k iteration [weights](https://lu.box.com/s/4dofbrcyz6yr9tmnmvdg62beoa8vbx1l) (sha1: 6727771807b0984f2f3bbed2cf4e0a2af80b396f).
3. Generate the fine-tuned weights for each split:
    - `run_generate_finetuned_instance_cache('<path-40k-instance-weights>', 'train', '<path-to-vgg19-weights>', 2000)`
    - `run_generate_finetuned_instance_cache('<path-40k-instance-weights>', 'val', '<path-to-vgg19-weights>', 2000)`
    - `run_generate_finetuned_instance_cache('<path-40k-instance-weights>', 'test', '<path-to-vgg19-weights>', 2000)`

#### 4. [Optional] Computing full-body OpenPose predictions

The official Caffe implementation of OpenPose supports predicting joints for face, hands and feet.
When running ACTOR in test time these can be used in the _3d reconstruction_ instead of the 15 joint representation.
To compute these features for Panoptic follow the steps below from the `openpose-caffe/` folder.

1. Clone the official [OpenPose repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose) into `openpose_caffe/openpose`.
2. To ensure same results checkout commit `1e4a7853572e491c5ec0afac4288346c9004065f`.
3. Build Caffe with Python support (see official documentation [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/1e4a7853572e491c5ec0afac4288346c9004065f/doc/installation.md) and [here](https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/1e4a7853572e491c5ec0afac4288346c9004065f/doc/modules/python_module.md)).
4. Install Python 3.5 and requirements in `requirements.txt` (or use Pipenv to do this automatically).
5. Generate the full-body cache by running:
    - `python cache_panoptic.py <data-folder-path>/train <cache-folder-path>/train`
    - `python cache_panoptic.py <data-folder-path>/val <cache-folder-path>/val`
    - `python cache_panoptic.py <data-folder-path>/test <cache-folder-path>/test`

## Additional information

Below follows in-depth information about the dataset and the scripts.

#### Explanation of scripts
  - `downloadAll.sh` downloads, extracts and, verifies all scenes.
  - `downloadScene.sh` downloads a scene with all videos and annoations
  - `extractScene.sh` extract images from the videos, removes videos and extracts the annotations. The videos frames are subsampled by a provided frequency. The annotations are then pruned to match the frames. Finally any Coco19 annotations are converted to MPII for all scenes, if they exists.
  - `subsample.sh` removes all but every n:th file in a directory.
  - `vgaImgsExtractor.sh` / `hdImgsExtractor.sh` extract image frames from the video then calls subsample on the resulting frames.
  - `verifyScene.sh` checks the content of the dataset.
  - `clean_up_panoptic.py` removes bad frames and frames missing annotations
  - `discard_annotations.py` removes annotations to match the subsampled frames.
  - `coco2mpii.py` converts the coco19 annotations to the MPII 15 joint format.
  - `openpose-tf/resize_merge.m` scales down the feature blobs and merges them into one file per scene.
  - `openpose-caffe/cache_panoptic.py` generates full-body 2d joint predictions instead of 15 joint version.

## Acknowledgements

  - Panoptic dataset scripts adapted from https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox
  - OpenPose TensorFlow implementation is from https://arvrjourney.com/human-pose-estimation-using-openpose-with-tensorflow-part-1-7dd4ca5c8027
