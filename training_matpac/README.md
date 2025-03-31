# MATPAC Training Code

In this folder we find what is required by fairseq framework to train a model.
We give the configuration file that was used to get the best model of our paper.

## Table of Contents

- [MATPAC Training Code](#matpac-training-code)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Data Preprocessing](#data-preprocessing)
  - [Training](#training)
  - [Configuration insights](#configuration-insights)

## Installation

- Firstly, install a conda environment using [environment.yml](environment.yml) file:

```bash
conda env create -f environment.yml
```

- Secondly, clone fairseq repository, use the same commit as we use for safety and reproducibility, then follow their installation guidelines:

```bash
git clone https://github.com/facebookresearch/fairseq.git

git checkout c214511
```

- Lastly, set an environment variable to point to your fairseq directory.
  
```bash
export FAIRSEQ=/path/to/fairseq
```


## Data Preprocessing

We use the scripts in [data_processing](./data_preprocessing/).

- For data processing first you should resample, mono and transform the audios into logmel spectrograms.

```python
python resample_and_lms.py --input_folder </path/that/contains/your/audios> --output_folder </path/that/will/contain/the/logmel/spectrograms> --sample_rate 16000 --n_mel 80 
```

- Secondly you need to create a 'manifest' file that list all the logmel spectrograms and their length.

```python
python prepare_manifest.py </path/that/will/contain/the/logmel/spectrograms> --dest </folder/where/you/save/the/manifest> --ext npy 
```

## Training

To launch a training use this command:

```python
python $FAIRSEQ/fairseq_cli/hydra_train.py -m --config-dir /path/to/matpac/training_matpac/config --config-name matpac common.user_dir /path/to/matpac/training_matpac task.data=/path/to/the/manifest
```

Alternatively to use a simpler command, you can overwrite in the [config file](./config/matpac.yaml):
- the `user_dir` under ``common``, which is the directory containing the fairseq-compliant model's code
- the `data` under `task`, which is the path to the manifest file.

And then the training command is only: 
```python
python $FAIRSEQ/fairseq_cli/hydra_train.py -m --config-dir /path/to/matpac/training_matpac/config --config-name matpac 
```

## Configuration insights

The config file [matpac.yaml](./config/matpac.yaml) has all the same hyperparameters as the best model of paper. However when training, we used a version of AudioSet where we had `2012615` samples above 6s.

As we presented several hyperparameters in epochs in our paper that are counted in steps in the config file, you will have to re-calculate their values in regards of your number of samples and your batch size.

The affected parameters are:

- ``lr_scheduler.warmup_updates``
- ``model.cls_head_cfg.warmup_n_steps``
- ``optimization.max_update``

i.e. for a 100000 samples, a batch size of 512 and a max update of 200 epochs you would have: 

```
max_update = (int(n_samples/bs)+1)*epochs
max_update = 39200
```