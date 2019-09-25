from pathlib import Path
import argparse

from src import dataframe

parser = argparse.ArgumentParser()
parser.add_argument("file", type=str, help="file to describe, shall be csv format")
parser.add_argument("-nh", "--no_header", action="store_true", help="The 1st line of the dataset is data not a header")
args = parser.parse_args()
df = dataframe.DataFrame()
try:
    df.read_from_csv(args.file, header=not args.no_header)
except (FileExistsError, FileNotFoundError, IsADirectoryError, PermissionError, NotADirectoryError, ValueError, IndexError, UnicodeDecodeError, UnicodeError, UnicodeEncodeError) as err:
    print("Could not read file '{}' because : {}".format(Path(args.file), err))
    exit(0)
df.describe()
