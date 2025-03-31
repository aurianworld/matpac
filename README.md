# MATPAC: Masked latent Prediction And Classification

**tl;dr**: MATPACK is powerful general audio fundation model.

This repository provides the training code, in fairseq framework, as well as an easy to use inference code from our paper [📝 Masked Latent Prediction and Classification for Self-Supervised Audio Representation Learning](https://ieeexplore.ieee.org/document/10887666)

![MATPAC](./assets/MATPAC.png)

## Table of Contents
- [MATPAC: Masked latent Prediction And Classification](#matpac-masked-latent-prediction-and-classification)
  - [Table of Contents](#table-of-contents)
  - [Inference code](#inference-code)
  - [Training code](#training-code)
  - [Benchmarks](#benchmarks)
  - [Citation](#citation)
  - [Credits](#credits)

## Inference code

The package [inference_matpac](./inference_matpac/) enables you to easily pip install the inference code to use MATPAC as a feature extractor from audio files.

```bash
pip install -e ./inference_matpac 
```

You can find more detail on how to use the inference code [here](./inference_matpac/README.md).

## Training code

The training code is available for reproducibility [here](./training_matpac/). We used [fairseq](https://github.com/facebookresearch/fairseq) as a training framework, therefore our code follows its guidelines which makes it easy to run. 
You can find a detailed explanation on how to run the model [here](./training_matpac/README.md).

## Benchmarks
MATPAC outperforms other self-supervised baselines on datasets such as OpenMIC, GTZAN, Magna-tag-a-tune, ESC-50 and US8K. It even outperforms supervised baselines on Magna-tag-a-tune.

![Bencmarks](./assets/table_results.png)


## Citation

If you use this work, please cite:
```bibtex
@INPROCEEDINGS{10887666,
  author={Quelennec, Aurian and Chouteau, Pierre and Peeters, Geoffroy and Essid, Slim},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Masked Latent Prediction and Classification for Self-Supervised Audio Representation Learning}, 
  year={2025},
  volume={},
  number={},
  pages={1-5},
  doi={10.1109/ICASSP49660.2025.10887666}}
```

---

## Credits

- [Fairseq](https://github.com/facebookresearch/fairseq) for the training framework. 
- [M2D](https://github.com/nttcslab/m2d/tree/master) for the base of the code.
- [DINO](https://github.com/facebookresearch/dino) for the classification head.

```
@inproceedings{M2D,
  author       = {Daisuke Niizumi and
                  Daiki Takeuchi and
                  Yasunori Ohishi and
                  Noboru Harada and
                  Kunio Kashino},
  title        = {Masked Modeling Duo: Learning Representations by Encouraging Both
                  Networks to Model the Input},
  booktitle    = {{IEEE} International Conference on Acoustics, Speech and Signal Processing
                  {ICASSP} 2023, Rhodes Island, Greece, June 4-10, 2023},
  pages        = {1--5},
  publisher    = {{IEEE}},
  year         = {2023},
  doi          = {10.1109/ICASSP49357.2023.10097236},
}

@inproceedings{DINO,
  author       = {Mathilde Caron and
                  Hugo Touvron and
                  Ishan Misra and
                  Herv{\'{e}} J{\'{e}}gou and
                  Julien Mairal and
                  Piotr Bojanowski and
                  Armand Joulin},
  title        = {Emerging Properties in Self-Supervised Vision Transformers},
  booktitle    = {2021 {IEEE/CVF} International Conference on Computer Vision, {ICCV}
                  2021, Montreal, QC, Canada, October 10-17, 2021},
  pages        = {9630--9640},
  publisher    = {{IEEE}},
  year         = {2021},
  doi          = {10.1109/ICCV48922.2021.00951},
}

```
