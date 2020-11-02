# Probing Multimodal Embeddings for Linguistic Properties: the Visual-Semantic Case

This repository contains material and code related to the COLING 2020-paper `Probing Multimodal Embeddings for Linguistic Properties: the Visual Semantic Case`.

## Contact

All contact regarding this repository or the related paper is pointed to `dali@cs.umu.se`.


# Instructions - under construction

The code runs on Pytorch and is partially based on [VSE++](https://github.com/fartashf/vsepp/).

## External dependencies

* VSE++
* VSE-C
* HAL
* MS-COCO

# Training and evaluation

```
usage: evaluate.py [-h] [--annotation_path ANNOTATION_PATH]
                   [--data_path DATA_PATH] [--split SPLIT]
                   [--model_path MODEL_PATH] [--embedding_path EMBEDDING_PATH]
                   [--resultfile RESULTFILE]

optional arguments:
  -h, --help            show this help message and exit
  --annotation_path ANNOTATION_PATH
  --data_path DATA_PATH
  --model_path MODEL_PATH
  --embedding_path EMBEDDING_PATH
  --resultfile RESULTFILE
```
