import argparse

parser = argparse.ArgumentParser(description="Traning options",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("dataset", help="Dataset location for .npz files")
parser.add_argument("-n", "--n-strokes", action="store", type=int,
                    help="the maximum number of sketch strokes", default=128)
parser.add_argument("-t", "--train-samples", action="store", type=int,
                    help="the number of samples for train set per class", default=5000)
parser.add_argument("-v", "--valid-samples", action="store", type=int,
                    help="the number of samples for valid set per class", default=1000)
parser.add_argument("-c", "--classes", action="store", type=int,
                    help="the number of output classes", default=345)
parser.add_argument("-e", "--epochs", action="store", type=int, default=50)
parser.add_argument("-b", "--batch-size", action="store", type=int, default=64)
parser.add_argument("-l", "--lr", action="store", type=float,
                    help="learning rate", default=1e-3)
parser.add_argument("-m", "--model-name", action="store", help="model name")
parser.add_argument("--model-state", action="store",
                    help="model state location to continue")
parser.add_argument("--logs", action="store",
                    help="path to save logs", default="log.json")
parser.add_argument("--save-figures", action="store",
                    help="directory name to save figures", default="figures")
parser.add_argument("--wandb-api", action="store",
                    help="API token to use wandb, or use environment variable WANDB_API_KEY")
parser.add_argument("--batch-norm", action="store_true",
                    help="use batch normalization")
parser.add_argument("--cuda", action="store", type=int,
                    help="cuda number to use", default=0)
parser.add_argument("--progress-bar", action="store_true",
                    help="Show progress bar")

args = parser.parse_args()

# as dict
config = vars(args)
print(config)
