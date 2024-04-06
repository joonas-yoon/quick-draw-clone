# Dataset

can get easliy from python package by [quickdraw API](https://quickdraw.readthedocs.io/en/latest/index.html)

# Train

```
train_cnn.py -n 4000 -e 100 -c class-list-full.txt --lr 0.01 -b 1024 --dropout 0.15 --checkpoint "checkpoints/epoch=50-step=1097100.ckpt"
```
