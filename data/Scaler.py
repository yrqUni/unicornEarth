import os 
import joblib
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="unicornEarth")
parser.add_argument('--inputPath', type=str, default='../DATA/RAW/', help='')
parser.add_argument('--outPath', type=str, default='../DATA/Scale/', help='')
parser.add_argument('--ScalerModel', type=str, default=None, help='')
parser.add_argument('--var', type=str, default=None, help='')
args = parser.parse_args()

if args.ScalerModel==None:
    args.ScalerModel = f'./SM/{args.var}'
ScalerModel = joblib.load(args.ScalerModel)
def StandardScaler(x):
    x = (x-ScalerModel['MeanAll'])/ScalerModel['stdAll']
    return x

dataPaths = []
for root, dirs, files in os.walk(args.inputPath, topdown=False):
    for file in files:
        if file==args.var:
            dataPaths.append(os.path.join(root, file))

for dataPath in tqdm(dataPaths):
    data = joblib.load(dataPath)
    data = StandardScaler(data)
    os.makedirs(os.path.abspath(os.path.dirname(f'{args.outPath}{dataPath[-(len(args.var)+5):]}')), exist_ok=True)
    # print(f'{args.outPath}{dataPath[-(len(args.var)+5):]}')
    joblib.dump(data,f'{args.outPath}{dataPath[-(len(args.var)+5):]}')

print('DONE!!!')