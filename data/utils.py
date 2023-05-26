import numpy as np
import datetime

def id2position(node_id, lat_len, lon_len):
    lat = node_id // lon_len
    lon = node_id % lon_len
    cos_lat = np.cos((0.5 - (lat + 1) / (lat_len + 1)) * np.pi)
    sin_lon = np.sin(lon / lon_len * np.pi)
    cos_lon = np.cos(lon / lon_len * np.pi)
    return cos_lat, sin_lon, cos_lon

def fetch_time_features(cursor_time):

    year_hours = (datetime.date(cursor_time.year + 1, 1, 1) - datetime.date(cursor_time.year, 1, 1)).days * 24
    next_year_hours = (datetime.date(cursor_time.year + 2, 1, 1) - datetime.date(cursor_time.year + 1, 1, 1)).days * 24

    cur_hour = (cursor_time - datetime.datetime(cursor_time.year, 1, 1)) / datetime.timedelta(hours=1)
    time_features = []
    for j in range(1440):
        # local time
        local_hour = cur_hour + j * 24 / 1440
        if local_hour > year_hours:
            tr = (local_hour - year_hours) / next_year_hours
        else:
            tr = local_hour / year_hours

        time_features.append([[np.sin(2 * np.pi * tr), np.cos(2 * np.pi * tr)]] * 720)

    return np.transpose(np.asarray(time_features), [1, 0, 2])


def fetch_constant_features():
    constant_features = []
    for i in range(720):
        tmp = []
        for j in range(1440):
            tmp.append(id2position(i * 1440 + j, 720, 1440))
        constant_features.append(tmp)
    return np.asarray(constant_features)


# a = datetime.datetime(2000,1,1,18)
# b = fetch_time_features(a)
# print(b.shape,b.max(),b.min())
# a = fetch_constant_features()
# print(a.shape)