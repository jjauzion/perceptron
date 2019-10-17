from pathlib import Path
import argparse

from src import processing

parser = argparse.ArgumentParser()
parser.add_argument("file", help="model file path")
args = parser.parse_args()

if not Path(args.file).is_file():
    print("File '{}' can't be found".format(args.file))
    exit(0)
model = processing.NeuralNetwork()
if not model.load_model(args.file):
    print("Error while loading model...")
    exit(0)
model.describe()
