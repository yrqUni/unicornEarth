import joblib
len_data = 0
data_info = joblib.load('/public/home/hydeng/Workspace/yrqUni/unicornEarth/data/DataInfo')

for infos_key in data_info['sample']:
    infos = data_info['sample'][infos_key]
    len_data = len_data+infos[0]
len_train_dataloader = int(len_data*(1-0.1))
print(len_train_dataloader)