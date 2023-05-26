import os 
import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="unicornEarth")
parser.add_argument('--inputPath', type=str, default=None, help='')
parser.add_argument('--outPath', type=str, default=None, help='')
parser.add_argument('--outPathMask', type=str, default=None, help='')
parser.add_argument('--size', type=int, default=6, help='')
parser.add_argument('--Target', type=str, default='TCWV', help='')
parser.add_argument('--UseValue', nargs='*', default=['TCWV','U10','V10','T2','MSLP','SP',
                                                    'VV100','U100','V100','RH100','T100',
                                                    'VV300','U300','V300','RH300','T300',
                                                    'VV500','U500','V500','RH500','T500',
                                                    'VV850','U850','V850','RH850','T850',
                                                    'VV1000','U1000','V1000','RH1000','T1000',
                                                    'TIME1','TIME2','POS1','POS2','POS3',], help='')
args = parser.parse_args()

fileName = [i for i in os.listdir(args.inputPath)]
fileName.remove(args.Target)
fileName.insert(0,args.Target)
tmp = []
tmpP = []
for i in tqdm(fileName):
    if i in args.UseValue:
        P = os.path.join(args.inputPath,f'{i}')
        x = joblib.load(P)
        x = sliding_window_view(x,2,0).transpose(0,3,1,2)
        tmp.append(x)
        tmpP.append(i)
tmp.reverse()
tmpP.reverse()
pad = np.zeros_like(tmp[0])
tmpi = []
tmpi_ = []
for i in tqdm(range(int(args.size))):
    tmpj = []
    tmpj_ = []
    for j in range(args.size):
        try:
            tmpj.append(tmp.pop())
            tmpj_.append(1)
        except:
            tmpj.append(pad)
            tmpj_.append(0)
            print(i,j,pad.sum())
    tmpi.append(np.concatenate(tmpj,axis=2))
    tmpi_.append(np.asarray(tmpj_))
tmp = np.concatenate(tmpi,axis=3)
tmp_ = np.asarray(tmpi_)
print(tmp.shape)
print(tmp_.shape)
# plt.matshow(tmp[0,:,:])
# plt.savefig('./b.jpg')
os.makedirs(os.path.abspath(os.path.dirname(args.outPath)), exist_ok=True)
os.makedirs(os.path.abspath(os.path.dirname(f'{args.outPathMask}')), exist_ok=True)
joblib.dump(tmp,f'{args.outPath}')
joblib.dump(tmp_,f'{args.outPathMask}')
print('DONE!!!')
