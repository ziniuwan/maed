Throughout the documentation we refer to MAED root folder as `$ROOT`. All the datasets listed below should be put in or linked to `$ROOT/data`. 

# Data Preparation

## 1. Download Datasets
You should first down load the datasets used in MAED.

- **InstaVariety**

Download the
[preprocessed tfrecords](https://github.com/akanazawa/human_dynamics/blob/master/doc/insta_variety.md#pre-processed-tfrecords) 
provided by the authors of Temporal HMR.

Directory structure:
```shell script
insta_variety
|-- train
|   |-- insta_variety_00_copy00_hmr_noS5.ckpt-642561.tfrecord
|   |-- insta_variety_01_copy00_hmr_noS5.ckpt-642561.tfrecord
|   `-- ...
`-- test
    |-- insta_variety_00_copy00_hmr_noS5.ckpt-642561.tfrecord
    |-- insta_variety_01_copy00_hmr_noS5.ckpt-642561.tfrecord
    `-- ...
```

As the original InstaVariety is saved in tfrecord format, which is not suitable for use in Pytorch. You could run this 
[script](../scripts/prepare_insta.sh) which will extract frames of every tfrecord and save them as jpeg.

Directory structure after extraction:
```shell script
insta_variety_img
|-- train
    |-- insta_variety_00_copy00_hmr_noS5.ckpt-642561.tfrecord
    |   |-- 0
    |   |-- 1
    |   `-- ...
    |-- insta_variety_01_copy00_hmr_noS5.ckpt-642561.tfrecord
    |   |-- 0
    |   |-- 1
    |   `-- ...
    `-- ...
```

- **[MPI-3D-HP](http://gvv.mpi-inf.mpg.de/3dhp-dataset)**

Donwload the dataset using the bash script provided by the authors. We will be using standard cameras only, so wall and ceiling
cameras aren't needed. Then, run  
[the script from the official VIBE repo](https://gist.github.com/mkocabas/cc6fe78aac51f97859e45f46476882b6) to extract frames of videos.

Directory structure:
```shell script
$ROOT/data
mpi_inf_3dhp
|-- S1
|   |-- Seq1
|   |-- Seq2
|-- S2
|   |-- Seq1
|   |-- Seq2
|-- ...
`-- util
```

- **[Human 3.6M](http://vision.imar.ro/human3.6m/description.php)**

Human 3.6M is not a open dataset now, thus it is optional in our training code. **However, Human 3.6M has non-negligible effect on the final performance of MAED.** 

Once getting available to the Human 3.6M dataset, one could refer to [the script](https://github.com/nkolot/SPIN/blob/master/datasets/preprocess/h36m_train.py) from the official SPIN repository to preprocess the Human 3.6M dataset.
Directory structure: 
```shell script
human3.6m
|-- annot
|-- dataset_extras
|-- S1
|-- S11
|-- S5
|-- S6
|-- S7
|-- S8
`-- S9
```

- **[3DPW](https://virtualhumans.mpi-inf.mpg.de/3DPW)**

Directory structure: 
```shell script
3dpw
|-- imageFiles
|   |-- courtyard_arguing_00
|   |-- courtyard_backpack_00
|   |-- ...
`-- sequenceFiles
    |-- test
    |-- train
    `-- validation
```

- **[PennAction](http://dreamdragon.github.io/PennAction/)** 

Directory structure: 
```shell script
pennaction
|-- frames
|   |-- 0000
|   |-- 0001
|   |-- ...
`-- labels
    |-- 0000.mat
    |-- 0001.mat
    `-- ...
```

- **[PoseTrack](https://posetrack.net/)** 

Directory structure: 
```shell script
posetrack
|-- images
|   |-- train
|   |-- val
|   |-- test
`-- posetrack_data
    `-- annotations
        |-- train
        |-- val
        `-- test
```

- **[MPII](http://human-pose.mpi-inf.mpg.de/)**

Directory structure: 
```shell script
mpii
|-- 099992483.jpg
|-- 099990098.jpg
`-- ...
```

- **[COCO 2014-All](https://cocodataset.org/)**

Directory structure: 
```shell script
coco2014-all
|-- 099992483.jpg
|-- 099990098.jpg
`-- ...
```

- **[LSPet](https://sam.johnson.io/research/lspet.html)**

Directory structure: 
```shell script
lspet
|-- COCO_train2014_000000000001.jpg
|-- COCO_train2014_000000000002.jpg
`-- ...
```

## 2. Download Annotation (pt format)
Download annotation data for MAED from [Google Drive](https://drive.google.com/drive/folders/1vApUaFNqo-uNP7RtVRxBy2YJJ1IprnQ8?usp=sharing) and move the whole directory to `$ROOT/data`.

## 3. Download SMPL data
Download SMPL data for MAED from [Google Drive](https://drive.google.com/drive/folders/1RqkUInP_0DohMvYpnFpqo7z_KWxjQVa6?usp=sharing) and move the whole directory to `$ROOT/data`.

## It's Done!
After downloading all the datasets and annotations, the directory structure of `$ROOT/data` should be like:
```shell script
$ROOT/data
|-- insta_variety
|-- insta_variety_img
|-- 3dpw
|-- mpii3d
|-- posetrack
|-- pennaction
|-- coco2014-all
|-- lspet
|-- mpii
|-- smpl_data
    |-- J_regressor_extra.npy
    `-- ...
`-- database
    |-- insta_train_db.pt
    |-- 3dpw_train_db.pt
    |-- lspet_train_db.pt
    `-- ...
```
