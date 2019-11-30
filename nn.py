import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor
from datetime import datetime
from geopy.geocoders import Nominatim
import numpy as np
def ParseCloudiness(x):
    if(x == "no clouds" or x == "" or x == "" or pd.isna(x)):
        return 0
    if(x == "Sky obscured by fog and/or other meteorological phenomena."):
        return 100
    x = str(x).replace('\xc2', ' ')
    x = x.replace('\x96', ' ')
    x = x.replace('%.', '')
    x = x.replace('%', '')
    nums = [int(s) for s in x.split() if s.isdigit()]
    if(len(nums) == 1):
        return str(nums[0])
    if(len(nums) == 2):
        return str((nums[0]+nums[1])/2.0)

def ParseWindDirection(x):
    if(x == "Calm, no wind" or x == "variable wind direction" or x == "" or pd.isna(x)):
        return 0
    x = str(x)
    north = x.count('north')
    east = x.count('east')
    south = x.count('south')
    west = x.count('west')
    all = []
    if(north != 0):
        all.extend(['north']*north)
    if(west != 0):
        all.extend(['west']*west)
    if(south != 0):
        all.extend(['south']*south)
    if(east != 0):
        all.extend(['east']*east)
    edge = GetDirectionEdge(all[0], 1)
    for i in range(1, len(all)):
        if(edge == GetDirectionEdge(all[i], edge)):
            continue
        elif(edge < GetDirectionEdge(all[i], edge)):
            edge += 90/(2**i)
        else:
            edge -= 90/(2**i)
    if(edge < 0):
        edge += 360
    return edge

    
def GetDirectionEdge(x, edge):
    if(x == 'north'): return 0
    if(x == 'east'): return 90
    if(x == 'south'): return 180
    if(x == 'west'): 
        if(edge == 0):
            return -90
        return 270

def ParseVisibility(x):
    if(x == "less than 0.1" or x == "less than 0.05"): return 0
    return str(x)

def TransformDate(x):
    dt = datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return dt.timetuple().tm_yday*24+dt.hour
    
class LocationSystem:
    def __init__(self):
        self.cities = {}

    def GetLocation(self, x):
        geolocator = Nominatim(user_agent="specify_your_app_name_here")
        location = geolocator.geocode(x)
        self.cities[x] = location

    def TransformToLatitude(self, x):
        if x in self.cities:
            location = self.cities[x]
            return location.latitude
        self.GetLocation(x)
        return self.TransformToLatitude(x)

    def TransformToLongitude(self, x):
        if x in self.cities:
            location = self.cities[x]
            return location.longitude
        self.GetLocation(x)
        return self.TransformToLatitude(x)

def CreateSet(csv):
    data = pd.read_csv(csv, names=names, sep=";", skiprows=[0], encoding='utf-8')

    locations = LocationSystem()

    data['Cloudiness'] = data['Cloudiness'].apply(ParseCloudiness)
    data['Wind direction'] = data['Wind direction'].apply(ParseWindDirection)
    data['Local time'] = data['Local time'].apply(TransformDate)
    data['Horizontal Visibility'] = data['Horizontal Visibility'].apply(ParseVisibility)
    data['Longitude'] = data['Latitude']
    data['Latitude'] = data['Latitude'].apply(locations.TransformToLatitude)
    data['Longitude'] = data['Longitude'].apply(locations.TransformToLongitude)
    data = data.astype('float64')
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    return data


names = ['Local time', 'Temperature', 'Pressure (station)', 'Pressure (sea level)', 'Humidity', 'Wind direction',
 'Wind m/s', 'Cloudiness', 'Horizontal Visibility',  'Dewpoint temperature', 'Latitude']

train_data = CreateSet('C:/Users/Mateusz/source/repos/SN_Project3/data/train_1')

X_train = train_data.loc[:,train_data.columns != 'Temperature']
Y_train = train_data['Temperature']

scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)
n2 = [n for n in names if n != 'Temperature']
n2.append('Longitude')

X_train = pd.DataFrame(X_train, columns=n2)

clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 20), random_state=1)
clf.fit(X_train,Y_train)

test_data = CreateSet('C:/Users/Mateusz/source/repos/SN_Project3/data/test_1')

X_test = test_data.loc[:,test_data.columns != 'Temperature']
Y_test = test_data['Temperature']
X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=n2)

predictions = clf.predict(X_test)

print(predictions-Y_test)