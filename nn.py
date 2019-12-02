import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor
import datetime
from geopy.geocoders import Nominatim
import numpy as np
import os
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
    dt = datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    return dt
def TransformDateFull(x):
    return x.timetuple().tm_yday*24+x.hour
    
class LocationSystem:
    def __init__(self):
        if(os.path.isfile('dictionary.npy')):
            self.cities = np.load('dictionary.npy',allow_pickle='TRUE').item()
        else:
            self.cities = {}

    def GetLocation(self, x):
        geolocator = Nominatim(user_agent="specify_your_app_name_here")
        location = geolocator.geocode(x)
        self.cities[x] = location
        np.save('dictionary.npy', self.cities) 

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

def WindToBool(x):
    if(x >= 15): return True
    return False

def MakeDateZero(x):
    return x.replace(hour=0, minute=0, second=0, microsecond=0)

def CreateSet(csv):
    data = pd.read_csv(csv, names=names, sep=";", skiprows=[0], encoding='utf-8')
    X = pd.DataFrame()
    Y = pd.DataFrame()

    locations = LocationSystem()

    data['Cloudiness'] = data['Cloudiness'].apply(ParseCloudiness)
    data['Wind direction'] = data['Wind direction'].apply(ParseWindDirection)
    data['Local time'] = data['Local time'].apply(TransformDate)
    data['Horizontal Visibility'] = data['Horizontal Visibility'].apply(ParseVisibility)
    data['Longitude'] = data['Latitude']
    data['Latitude'] = data['Latitude'].apply(locations.TransformToLatitude)
    data['Longitude'] = data['Longitude'].apply(locations.TransformToLongitude)
    #
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    latitude = 6.921812
    longitude = 79.865561
    eps = 0.01

    dates = pd.DataFrame(data['Local time'].apply(MakeDateZero).unique(), columns=['Local time']).sort_values(by='Local time').reset_index(drop=True)
    
    for index, row in dates.iterrows():
        curr_date = row['Local time']
        mask = (data['Local time'] > str(curr_date)) & (data['Local time'] < str(curr_date+datetime.timedelta(days=5))) & (abs(data['Latitude']-latitude) < eps) & (abs(data['Longitude']-longitude) < eps)
        maskY = (data['Local time'] > str(curr_date+datetime.timedelta(days=6))) & (data['Local time'] < str(curr_date+datetime.timedelta(days=7))) & (abs(data['Latitude']-latitude) < eps) & (abs(data['Longitude']-longitude) < eps)
        onerow = data.loc[mask].copy()
        onerowY = data.loc[maskY].copy()
        onerowY = onerowY[['Temperature', 'Wind m/s']]

        onerow['Local time'] = data['Local time'].apply(TransformDateFull)
        onerow = onerow.astype('float64')
        onerowY = onerowY.astype('float64')

        #todo interpolation of data
        onerow = onerow.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
        onerowY = onerowY.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

        if(onerow.shape[0] != 40 or onerowY.shape[0] != 8): # todo: interpolation of data
            continue
        
        onerow = onerow.reset_index(drop=True)
        onerow.index = onerow.index + 1
        onerow_out = onerow.stack()
        onerow_out.index = onerow_out.index.map('{0[1]}_{0[0]}'.format)
        onerow = onerow_out.to_frame().T

        onerowY = onerowY.reset_index(drop=True)
        onerowY.index = onerowY.index + 1
        onerowY_out = onerowY.stack()
        onerowY_out.index = onerowY_out.index.map('{0[1]}_{0[0]}'.format)
        onerowY = onerowY_out.to_frame().T

        X = X.append(onerow, ignore_index = True)
        Y = Y.append(onerowY, ignore_index = True)
    return [X, Y]


names = ['Local time', 'Temperature', 'Pressure (station)', 'Pressure (sea level)', 'Humidity', 'Wind direction',
 'Wind m/s', 'Cloudiness', 'Horizontal Visibility',  'Dewpoint temperature', 'Latitude']

[X_train, Y_train] = CreateSet('C:/Users/Mateusz/source/repos/SN_Project3/data/train_1')
n2 = X_train.columns.values
n2_Y = Y_train.columns.values
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_train = pd.DataFrame(X_train, columns=n2)
X_train.to_csv('trainX.csv')
Y_train.to_csv('trainY.csv')
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(30, 10), random_state=1)
#clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1, 1), random_state=1, verbose=1)
clf.fit(X_train,Y_train)

[X_test, Y_test] = CreateSet('C:/Users/Mateusz/source/repos/SN_Project3/data/test_1')

X_test = scaler.transform(X_test)
X_test = pd.DataFrame(X_test, columns=n2)

predictions = clf.predict(X_test)

predictions = pd.DataFrame(predictions, columns=n2_Y)
#print(predictions)
#print(Y_test)
predictions.loc[:, predictions.columns.str.startswith('Wind m/s')] = predictions.loc[:, predictions.columns.str.startswith('Wind m/s')].applymap(WindToBool)
Y_test.loc[:, Y_test.columns.str.startswith('Wind m/s')] = Y_test.loc[:, Y_test.columns.str.startswith('Wind m/s')].applymap(WindToBool)

good_predictions = predictions.loc[:, predictions.columns.str.startswith('Wind m/s')] == Y_test.loc[:, Y_test.columns.str.startswith('Wind m/s')]
good_predictions = good_predictions[good_predictions == True].sum().sum()
errorsTemperature = abs(predictions.loc[:, predictions.columns.str.startswith('Temperature')]-Y_test.loc[:, Y_test.columns.str.startswith('Temperature')])
errorsTemperature.to_csv('errorsTemp.log')
print('Temperature\nAverage error: ' + str(np.average(errorsTemperature)) + 'Average std: ' + str(np.std(errorsTemperature).mean()))
print('Wind\nGood predictions: ' + str(good_predictions/(Y_test.shape[0]*Y_test.shape[1]/2)*100) + "%")