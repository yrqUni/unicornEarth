import os
import joblib
import argparse
import math
from tqdm import tqdm

parser = argparse.ArgumentParser(description="unicornEarth")
parser.add_argument('--inputPath', type=str, default='../DATA/RAW', help='')
parser.add_argument('--outPath', type=str, default='./ScalerModel', help='')
parser.add_argument('--var', type=str, default=None, help='')
args = parser.parse_args()

dataPaths = []
for root, dirs, files in os.walk(args.inputPath, topdown=False):
    for file in files:
        if file==args.var:
            dataPaths.append(os.path.join(root, file))
meanLs = []
sizeLs = []
for dataPath in tqdm(dataPaths):
    data = joblib.load(dataPath)
    meanLs.append(data.mean())
    sizeLs.append(data.size)
MeanAll = sum(meanLs)/len(meanLs)
SizeAll = sum(sizeLs)
SqrDiffSumLs = []
for dataPath in tqdm(dataPaths):
    data = joblib.load(dataPath)
    SqrDiffSumLs.append(((data-MeanAll)*(data-MeanAll)).sum())
stdAll = math.sqrt(sum(SqrDiffSumLs)/SizeAll)

os.makedirs(os.path.abspath(os.path.dirname(args.outPath)), exist_ok=True)
joblib.dump({'MeanAll':MeanAll,'stdAll':stdAll},args.outPath)

# def StandardScaler(x):
#     x = (x-MeanAll)/stdAll
#     return x
