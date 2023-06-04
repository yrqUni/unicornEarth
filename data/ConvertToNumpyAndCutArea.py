import pygrib
import netCDF4 as nc
import os
import joblib
from datetime import datetime
import numpy as np
from tqdm import tqdm
import gc
from utils import fetch_constant_features, fetch_time_features
import argparse

parser = argparse.ArgumentParser(description="unicornEarth")
parser.add_argument('--a', type=int, default=175, help='')
parser.add_argument('--b', type=int, default=440, help='')
parser.add_argument('--offset', type=int,default=128,help='')
parser.add_argument('--years', nargs='*', default=None, help='')
parser.add_argument('--month', nargs='*', default=None, help='')
parser.add_argument('--outputPath',type=str, default=None, help='')
args = parser.parse_args()

fileNameAll = {}
if args.month == None:
    if args.years == None:
        Years = [int(i) for i in os.listdir(f'/sharedata/dataset/realtime/SD005-ERA5/0p25/yearly')] 
    if args.years != None:
        Years = args.years
    Years.sort()
if args.month != None:
    Years = [args.month[0][:4],]
for Year in Years:
    if args.month == None:
        YearMonths = [int(i) for i in os.listdir(f'/sharedata/dataset/realtime/SD005-ERA5/0p25/yearly/{Year}')] 
    if args.month != None:
        YearMonths = args.month
    YearMonths.sort()
    for YearMonth in YearMonths: 
        YearsMonthDays = [int(i[:8]) for i in os.listdir(f'/sharedata/dataset/realtime/SD005-ERA5/0p25/yearly/{Year}/{YearMonth}')]
        YearsMonthDays = list(set(YearsMonthDays))
        for YearsMonthDay in YearsMonthDays:
            fileNameAll[YearsMonthDay] = []
        for YearsMonthDay in YearsMonthDays:
            if os.path.isfile(f'/sharedata/dataset/realtime/SD005-ERA5/0p25/yearly/{Year}/{YearMonth}/{YearsMonthDay}-pressure.grib'):
                filePath = f'/sharedata/dataset/realtime/SD005-ERA5/0p25/yearly/{Year}/{YearMonth}/{YearsMonthDay}-pressure.grib'
                fileNameAll[YearsMonthDay].append([filePath,'pressure','grib'])
            if os.path.isfile(f'/sharedata/dataset/realtime/SD005-ERA5/0p25/yearly/{Year}/{YearMonth}/{YearsMonthDay}-single.grib'):
                filePath = f'/sharedata/dataset/realtime/SD005-ERA5/0p25/yearly/{Year}/{YearMonth}/{YearsMonthDay}-single.grib'
                fileNameAll[YearsMonthDay].append([filePath,'single','grib'])
            if os.path.isfile(f'/sharedata/dataset/realtime/SD005-ERA5/0p25/yearly/{Year}/{YearMonth}/{YearsMonthDay}-single.nc'):
                filePath = f'/sharedata/dataset/realtime/SD005-ERA5/0p25/yearly/{Year}/{YearMonth}/{YearsMonthDay}-single.nc'
                fileNameAll[YearsMonthDay].append([filePath,'single','nc'])   
for YearsMonthDay in list(fileNameAll.keys()):
    if len(fileNameAll[YearsMonthDay])!=2:
        del fileNameAll[YearsMonthDay]
        gc.collect()
fileNameAll = dict(sorted(fileNameAll.items(),key=lambda x:x[0]))
print(list(fileNameAll.keys())[-32:])

DATA = {'TCWV':[], 'U10':[], 'V10':[], 'T2':[], 'MSLP':[], 'SP':[],
        'VV100':[], 'U100':[], 'V100':[], 'RH100':[], 'T100':[],
        'VV300':[], 'U300':[], 'V300':[], 'RH300':[], 'T300':[],
        'VV500':[], 'U500':[], 'V500':[], 'RH500':[], 'T500':[],
        'VV850':[], 'U850':[], 'V850':[], 'RH850':[], 'T850':[],
        'VV1000':[], 'U1000':[], 'V1000':[], 'RH1000':[], 'T1000':[],
        'TIME1':[], 'TIME2':[], 'POS1':[], 'POS2':[], 'POS3':[]} 

P = fetch_constant_features()

for YearsMonthDay in tqdm(list(fileNameAll.keys())):
    _DATA = {'TCWV':[],'U10':[],'V10':[],'T2':[],'MSLP':[],'SP':[],
            'VV100':[], 'U100':[], 'V100':[], 'RH100':[],'T100':[],
            'VV300':[], 'U300':[], 'V300':[], 'RH300':[],'T300':[],
            'VV500':[], 'U500':[], 'V500':[], 'RH500':[],'T500':[],
            'VV850':[], 'U850':[], 'V850':[], 'RH850':[],'T850':[],
            'VV1000':[], 'U1000':[], 'V1000':[], 'RH1000':[],'T1000':[],
            'TIME1':[], 'TIME2':[], 'POS1':[], 'POS2':[], 'POS3':[]} 
    E = 0
    for filePath in fileNameAll[YearsMonthDay]:
        if filePath[1] == 'pressure':
            try:
                grbs = pygrib.open(filePath[0])
                t = datetime(3000,1,1,0)
                F = 0
                for grb in grbs:
                    if grb.parameterName == 'Relative humidity' and grb.level == 100:
                        _DATA['RH100'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        T_ = grb.validDate
                        T = fetch_time_features(T_)
                        if T_!=t or F==0:
                            _DATA['TIME1'].append(np.squeeze(T[args.a:args.a+args.offset,args.b:args.b+args.offset,0]))
                            _DATA['TIME2'].append(np.squeeze(T[args.a:args.a+args.offset,args.b:args.b+args.offset,1]))
                            _DATA['POS1'].append(np.squeeze(P[args.a:args.a+args.offset,args.b:args.b+args.offset,0]))
                            _DATA['POS2'].append(np.squeeze(P[args.a:args.a+args.offset,args.b:args.b+args.offset,1]))
                            _DATA['POS3'].append(np.squeeze(P[args.a:args.a+args.offset,args.b:args.b+args.offset,2]))
                            t = T_
                            F = 1
                    if grb.parameterName == 'Relative humidity' and grb.level == 300:
                        _DATA['RH300'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Relative humidity' and grb.level == 500:
                        _DATA['RH500'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Relative humidity' and grb.level == 850:
                        _DATA['RH850'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Relative humidity' and grb.level == 1000:
                        _DATA['RH1000'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])

                    if grb.parameterName == 'Temperature' and grb.level == 100:
                        _DATA['T100'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Temperature' and grb.level == 300:
                        _DATA['T300'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Temperature' and grb.level == 500:
                        _DATA['T500'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Temperature' and grb.level == 850:
                        _DATA['T850'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Temperature' and grb.level == 1000:
                        _DATA['T1000'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])

                    if grb.parameterName == 'U component of wind' and grb.level == 100:
                        _DATA['U100'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'U component of wind' and grb.level == 300:
                        _DATA['U300'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'U component of wind' and grb.level == 500:
                        _DATA['U500'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'U component of wind' and grb.level == 850:
                        _DATA['U850'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'U component of wind' and grb.level == 1000:
                        _DATA['U1000'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])

                    if grb.parameterName == 'V component of wind' and grb.level == 100:
                        _DATA['V100'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'V component of wind' and grb.level == 300:
                        _DATA['V300'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'V component of wind' and grb.level == 500:
                        _DATA['V500'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'V component of wind' and grb.level == 850:
                        _DATA['V850'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'V component of wind' and grb.level == 1000:
                        _DATA['V1000'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])

                    if grb.parameterName == 'Vertical velocity' and grb.level == 100:
                        _DATA['VV100'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Vertical velocity' and grb.level == 300:
                        _DATA['VV300'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Vertical velocity' and grb.level == 500:
                        _DATA['VV500'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Vertical velocity' and grb.level == 850:
                        _DATA['VV850'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                    if grb.parameterName == 'Vertical velocity' and grb.level == 1000:
                        _DATA['VV1000'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
            except:
                E = 1
                break    
        if filePath[1] == 'single':
            if filePath[2] == 'grib':
                try:
                    grbs = pygrib.open(filePath[0])
                    for grb in grbs:
                        if grb.parameterName == '10 metre U wind component' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,0))[-8:]:
                            _DATA['U10'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '10 metre U wind component' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,6))[-8:]:
                            _DATA['U10'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '10 metre U wind component' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,12))[-8:]:
                            _DATA['U10'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '10 metre U wind component' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,18))[-8:]:
                            _DATA['U10'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '10 metre V wind component' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,0))[-8:]:
                            _DATA['V10'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '10 metre V wind component' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,6))[-8:]:
                            _DATA['V10'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '10 metre V wind component' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,12))[-8:]:
                            _DATA['V10'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '10 metre V wind component' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,18))[-8:]:
                            _DATA['V10'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '2 metre temperature' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,0))[-8:]:
                            _DATA['T2'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '2 metre temperature' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,6))[-8:]:
                            _DATA['T2'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '2 metre temperature' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,12))[-8:]:
                            _DATA['T2'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == '2 metre temperature' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,18))[-8:]:
                            _DATA['T2'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Mean sea level pressure' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,0))[-8:]:
                            _DATA['MSLP'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Mean sea level pressure' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,6))[-8:]:
                            _DATA['MSLP'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Mean sea level pressure' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,12))[-8:]:
                            _DATA['MSLP'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Mean sea level pressure' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,18))[-8:]:
                            _DATA['MSLP'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Surface pressure' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,0))[-8:]:
                            _DATA['SP'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Surface pressure' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,6))[-8:]:
                            _DATA['SP'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Surface pressure' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,12))[-8:]:
                            _DATA['SP'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Surface pressure' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,18))[-8:]:
                            _DATA['SP'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Total column water vapour' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,0))[-8:]:
                            _DATA['TCWV'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Total column water vapour' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,6))[-8:]:
                            _DATA['TCWV'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Total column water vapour' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,12))[-8:]:
                            _DATA['TCWV'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                        if grb.parameterName == 'Total column water vapour' and str(grb.validDate)[-8:] == str(datetime(2000,1,1,18))[-8:]:
                            _DATA['TCWV'].append(np.array(grb.values)[args.a:args.a+args.offset,args.b:args.b+args.offset])
                except:
                    E = 1
                    break

        if filePath[2] == 'nc':
            try:
                f = nc.Dataset(filePath[0])
                _DATA['U10'].append(np.array(f.variables['u10'][0])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['U10'].append(np.array(f.variables['u10'][6])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['U10'].append(np.array(f.variables['u10'][12])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['U10'].append(np.array(f.variables['u10'][18])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['V10'].append(np.array(f.variables['v10'][0])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['V10'].append(np.array(f.variables['v10'][6])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['V10'].append(np.array(f.variables['v10'][12])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['V10'].append(np.array(f.variables['v10'][18])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['T2'].append(np.array(f.variables['t2m'][0])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['T2'].append(np.array(f.variables['t2m'][6])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['T2'].append(np.array(f.variables['t2m'][12])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['T2'].append(np.array(f.variables['t2m'][18])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['MSLP'].append(np.array(f.variables['msl'][0])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['MSLP'].append(np.array(f.variables['msl'][6])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['MSLP'].append(np.array(f.variables['msl'][12])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['MSLP'].append(np.array(f.variables['msl'][18])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['SP'].append(np.array(f.variables['sp'][0])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['SP'].append(np.array(f.variables['sp'][6])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['SP'].append(np.array(f.variables['sp'][12])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['SP'].append(np.array(f.variables['sp'][18])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['TCWV'].append(np.array(f.variables['tcwv'][0])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['TCWV'].append(np.array(f.variables['tcwv'][6])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['TCWV'].append(np.array(f.variables['tcwv'][12])[args.a:args.a+args.offset,args.b:args.b+args.offset])
                _DATA['TCWV'].append(np.array(f.variables['tcwv'][18])[args.a:args.a+args.offset,args.b:args.b+args.offset])
            except:
                E = 1
                break
    if E==0:
        tmp = []
        for K in list(_DATA.keys()):
            tmp.append(np.asarray(_DATA[K]).shape[0])
        if len(list(set(tmp))) == 1:
            for K in list(_DATA.keys()):
                DATA[K].append(np.asarray(_DATA[K]))

os.makedirs(os.path.abspath(os.path.dirname(args.outputPath)), exist_ok=True)
for K in tqdm(DATA):
    data = np.concatenate(DATA[K], axis=0)
    print(f'{K}:{data.shape}')
    joblib.dump(data,f'{args.outputPath}/{K}')

print("DONE!!!")

