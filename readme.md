# Transformer Text Diffusion


This repository contains an implementation of a Denoising Diffusion Probabilistic Model of Text (DDPT) based on Transformer networks. This model aims to generate high-quality, coherent text by utilizing diffusion-based probabilistic modeling techniques within a transformer architecture.

| ![diffusion](./generation.gif) |

## Setup

```bash
pip install -r requirements.txt
```


## Training the Model

1. First create the model
```bash
python create_model.py
```
2. edit `sh train.sh` with the desired parameters and run:
```bash
sh train.sh
```
## Usage

You can generate and play with the model with the provided jupyter notebook `generate sample.ipynb`

## Usage

This repository was built on top of code from [minimal-text-diffusion](https://github.com/madaan/minimal-text-diffusion), [diffusers](https://github.com/huggingface/diffusers) and [transformers](https://github.com/huggingface/transformers)