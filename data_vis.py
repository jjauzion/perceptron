from src import dataframe
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="dataset file path, shall be csv format")
args = parser.parse_args()

df = dataframe.DataFrame()
try:
    df.read_from_csv(args.file, header=False, converts={1: ["B", "M"]})
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
df.describe()
# df.pair_plot(hue_col=1)
df.scatter(ycol=1)
print("finish")
