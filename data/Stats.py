import os
import joblib
import argparse
import math
from tqdm import tqdm

parser = argparse.ArgumentParser(description="unicornEarth")
parser.add_argument('--inputPath', type=str, default='../DATA/Scale', help='')
parser.add_argument('--outPath', type=str, default='./ScalerModel', help='')
parser.add_argument('--var', type=str, default=None, help='')
args = parser.parse_args()

dataPaths = []
for root, dirs, files in os.walk(args.inputPath, topdown=False):
    for file in files:
        if file==args.var:
            dataPaths.append(os.path.join(root, file))
minLs = []
maxLs = []
for dataPath in tqdm(dataPaths):
    data = joblib.load(dataPath)
    minLs.append(data.min())
    maxLs.append(data.max())
minVal = min(minLs)
maxVal = max(maxLs)

os.makedirs(os.path.abspath(os.path.dirname(args.outPath)), exist_ok=True)
joblib.dump({'Min':minVal,'Max':maxVal},args.outPath)

# def StandardScaler(x):
#     x = (x-MeanAll)/stdAll
#     return x
