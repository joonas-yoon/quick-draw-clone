
# ## Importings
from datamodules import DataModule, QuickDrawDataAllSet
import argparse
import warnings
import os
import torch
from torchvision import transforms as T
import models
import gc
gc.enable()

print("pytorch version:", torch.__version__)

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description="Export options",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c", "--classes", action="store", type=str,
                    help="the list of classes to train")
parser.add_argument("-m", "--model-name", action="store", help="model name")
parser.add_argument("--checkpoint", action="store",
                    help="[lightning] model checkpoint")
parser.add_argument("--seed", type=int, default=1234,
                    metavar="S", help="random seed (default: 1234)")

config = parser.parse_args()

with open(config.classes, 'r') as f:
    CLASSES = sorted(filter(lambda s: bool(len(s)), f.readlines()))
    CLASSES = list(map(lambda s: s.replace('\n', ''), CLASSES))
    print('predict classes:', CLASSES)


# Horizontal line as divider
HR = "\n" + ("-" * 30) + "\n"

# Options
SEED = int(config.seed)

# - Dataset
OUT_CLASSES = len(CLASSES)

# - Load model
MODEL_CHECKPOINT = config.checkpoint
MODEL_OUTPUT_NAME = config.model_name or f'model'

# Set environment before importing torch
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

torch.manual_seed(SEED)


def get_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"  # Apple GPU
    return "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available


device = get_device_name()
print('device:', device, HR)

encode_image = T.Compose([
    T.Resize(32),
    T.ToTensor(),
    #   T.Normalize(mean=[0.485, 0.456, 0.406],
    #               std=[0.229, 0.224, 0.225]),
    T.Normalize(0.5, 0.5)
])
decode_image = T.Compose([
    T.Normalize(mean=[0, 0, 0], std=[1/0.5, 1/0.5, 1/0.5]),
    T.Normalize(mean=[-0.5, -0.5, -0.5], std=[1, 1, 1]),
    T.ToPILImage()
])

# ### Load dataset
dataset_all = QuickDrawDataAllSet(
    CLASSES, max_drawings=10, transform=encode_image)

print("The number of classes to predict:", OUT_CLASSES)

# ## Load model
model = models.CNNModel(OUT_CLASSES, dropout=0.0)
model.to(device)

print("Model: ===============================")
print(model, HR)

print("Load from checkpoint")
checkpoint = torch.load(config.checkpoint)
model.load_state_dict(checkpoint["state_dict"])

print("Prepare Dataloader to get shape of input")
dm = DataModule(dataset=dataset_all, batch_size=1)
x, y = next(iter(dm.test_dataloader()))

print("Export to ONNX")
model.to_onnx(file_path=f"{config.model_name}.onnx", input_sample=x)
