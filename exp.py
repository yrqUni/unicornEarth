# import joblib
# len_data = 0
# data_info = joblib.load('/public/home/hydeng/Workspace/yrqUni/unicornEarth/data/DataInfo')

# for infos_key in data_info['sample']:
#     infos = data_info['sample'][infos_key]
#     len_data = len_data+infos[0]
# len_train_dataloader = int(len_data*(1-0.1))
# print(len_train_dataloader)

# import numpy as np
# from sklearn.model_selection import train_test_split

# a = np.arange(0,100,1)
# a = a.reshape(25,4)
# X_train, X_test, y_train, y_test = train_test_split(a[:,:3], a[:,3:], test_size=0.33, random_state=1,shuffle=False)
# X_train, X_test, y_train, y_test = train_test_split(a[:,:3], a[:,3:], test_size=0.33, random_state=1,shuffle=False)
