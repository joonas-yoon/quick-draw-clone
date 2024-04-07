# Dataset

can get easliy from python package by [quickdraw API](https://quickdraw.readthedocs.io/en/latest/index.html)

# Train

```
python train_cnn.py -n 4000 -e 100 -c class-list-full.txt --lr 0.01 -b 1024 --dropout 0.15 --checkpoint "checkpoints/epoch=50-step=1097100.ckpt"
```

# Export to ONNX

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python export_to_onnx.py -c class-list-full.txt --checkpoint 3z4ficif.ckpt --model-name model-3z4ficif
```

## Trouble shooting

If you try to run on Apple Mac M1 or above, set environment variable and run to temporary fix:

```
PYTORCH_ENABLE_MPS_FALLBACK=1 python ...
```