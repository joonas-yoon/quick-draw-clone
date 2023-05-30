# Backend

## Classifier

Train neural net using LSTM to classify 345 labels from user drawing sketches

### Model

```
LitLSTM(
  (conv): Sequential(
    (0): Conv1d(256, 64, kernel_size=(3,), stride=(1,))
    (1): Dropout(p=0.2, inplace=False)
    (2): Conv1d(64, 96, kernel_size=(1,), stride=(1,))
    (3): Dropout(p=0.2, inplace=False)
    (4): Conv1d(96, 128, kernel_size=(1,), stride=(1,))
    (5): Dropout(p=0.2, inplace=False)
  )
  (lstm1): LSTM(128, 256, num_layers=2, batch_first=True, dropout=0.2)
  (lstm2): LSTM(256, 256, num_layers=2, dropout=0.2)
  (fc): Sequential(
    (0): ReLU(inplace=True)
    (1): Dropout(p=0.1, inplace=False)
    (2): Linear(in_features=256, out_features=512, bias=True)
    (3): ReLU(inplace=True)
    (4): Dropout(p=0.1, inplace=False)
    (5): Linear(in_features=512, out_features=345, bias=True)
  )
)
```

## Train with Lightning

Sample command to train:

```bash
python train_light.py -n 256 -c 345 \
    --train-samples 20000 --valid-samples 1000 \
    --epochs 100 --batch-size 8192 --lr 0.001 \
    --wandb-api <api-key> \
    dataset/sketches/sketches/
```

You can run help command to show usage:

```bash
python train_light.py --help
```

## Export to ONNX Model

```py
import torch
from models import LitLSTM
from onnx_functions import export_to_onnx

model = LitLSTM(256, 345)
ckpt = torch.load("./model.ckpt")
model.load_state_dict(ckpt['state_dict'])
export_to_onnx(model, "model.onnx", (1, 256, 3))
```
