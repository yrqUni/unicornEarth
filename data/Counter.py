import joblib
from tqdm import tqdm
count1 = {}
for year in tqdm(range(2000,2022)):
    data = joblib.load(f'../DATA/Merge/{year}')
    count1[year] = data.shape
count2 = {}
for year in tqdm(range(2000,2022)):
    data = joblib.load(f'../DATA/PadMask/{year}')
    count2[f'{year}'] = data.shape
count = {'sample':count1,'pad_mask':count2}
joblib.dump(count,f'DataInfo')
