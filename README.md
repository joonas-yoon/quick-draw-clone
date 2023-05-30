# Quick Draw Clone

## Model

### Dataset

**1. Download zip**

From kaggle by Google: https://www.kaggle.com/datasets/google/tinyquickdraw

- Download from link: https://www.kaggle.com/datasets/google/tinyquickdraw/download?datasetVersionNumber=3

**2. Extract zip file**

Extract donwloaded zip file with name `dataset/`

```bash
unzip quickdraw_simplified.zip -d dataset
```

There must be `dataset/sketches/sketches/<label>.npz` that are data what we want to train.

### Export

Export to [ONNX](https://github.com/onnx/onnx) as an interface to deploy on web application.

## Web UI

### CRA

Build with Create React Application

### ONNX.js

Use [ONNX.js](https://github.com/microsoft/onnxjs) to load and run ONNX models.
