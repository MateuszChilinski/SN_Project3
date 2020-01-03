import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.neural_network import MLPRegressor
import datetime
from geopy.geocoders import Nominatim
import numpy as np
import os
import math
import time 
names = ['Local time', 'Temperature', 'Pressure (station)', 'Pressure (sea level)', 'Humidity', 'Wind direction',
    'Wind m/s', 'Cloudiness', 'Horizontal Visibility',  'Dewpoint temperature', 'Latitude']
eps = 0.01
timestr = time.strftime("%Y%m%d-%H%M%S")

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

def ParseWinSin(x):
    return math.sin(x)

    
def GetDirectionEdge(x, edge):
    if(x == 'north'): return 0
    if(x == 'east'): return 90
    if(x == 'south'): return 180
    if(x == 'west'): 
        if(edge == 0):
            return -90
        return 270

def ParseVisibility(x):
    if(x == "less than 0.1" or x == "less than 0.05" or x == "" or pd.isna(x)): return 0
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
    
    def GetAllCities(self):
        return self.cities

def WindToBool(x):
    if(x >= 9): return True
    return False

def MakeDateZero(x):
    return x.replace(hour=0, minute=0, second=0, microsecond=0)

def CreateSet(csv, interpolate=0, applyWindTransformation=0):
    data = pd.read_csv(csv, names=names, sep=";", skiprows=[0], encoding='utf-8')
    X = pd.DataFrame()
    Y = pd.DataFrame()

    locations = LocationSystem()

    data['Cloudiness'] = data['Cloudiness'].apply(ParseCloudiness)
    data['Wind direction'] = data['Wind direction'].apply(ParseWindDirection)
    if(applyWindTransformation == 1):
        data['Wind direction'] = data['Wind direction'].apply(ParseWinSin)
    data['Local time'] = data['Local time'].apply(TransformDate)
    data['Horizontal Visibility'] = data['Horizontal Visibility'].apply(ParseVisibility)
    print("Sorting...", flush=True)
    data = data.sort_values(by=['Latitude', 'Local time'])

    data['Longitude'] = data['Latitude']
    data['Latitude'] = data['Latitude'].apply(locations.TransformToLatitude)
    data['Longitude'] = data['Longitude'].apply(locations.TransformToLongitude)
    #
    data = data.reset_index(drop=True)
    if(interpolate == 0):
        print("Droping NA...", flush=True)
        data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)
    else:
        print("Interpolating...", flush=True) # interpolate data
        data = data.interpolate(method='nearest', axis=0).ffill().bfill()
    data = data.reset_index(drop=True)

    dates = pd.DataFrame(data['Local time'].apply(MakeDateZero).unique(), columns=['Local time']).sort_values(by='Local time').reset_index(drop=True)
    cities = locations.GetAllCities()
    i = 0
    for city in locations.GetAllCities():
        print("Cities done: " + str(i) + "/" + str(len(cities)))
        i = i+1
        latitude = locations.TransformToLatitude(city)
        longitude = locations.TransformToLongitude(city)
        for index, row in dates.iterrows():
            curr_date = row['Local time']
            mask = (data['Local time'] > str(curr_date)) & (data['Local time'] < str(curr_date+datetime.timedelta(days=5))) & (abs(data['Latitude']-latitude) < eps) & (abs(data['Longitude']-longitude) < eps)
            maskY = (data['Local time'] > str(curr_date+datetime.timedelta(days=6))) & (data['Local time'] < str(curr_date+datetime.timedelta(days=7))) & (abs(data['Latitude']-latitude) < eps) & (abs(data['Longitude']-longitude) < eps)
            onerow = data.loc[mask].copy()
            onerowY = data.loc[maskY].copy()
            if(onerow.shape[0] != 40 or onerowY.shape[0] != 8): # there is not enough data
                continue
            onerowY = onerowY[['Temperature', 'Wind m/s']]

            onerow['Local time'] = data['Local time'].apply(TransformDateFull)
            onerow = onerow.astype('float64')
            onerowY = onerowY.astype('float64')

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

            X = X.append(onerow, ignore_index = True, sort=False)
            Y = Y.append(onerowY, ignore_index = True, sort=False)
    print("Data prepared...", flush=True)
    return [X, Y]

def CreateTestinScenario(name, train, test, architecture, interpolate=0, applyWindTransformation=0):
    [X_train, Y_train] = CreateSet(train, interpolate, applyWindTransformation)
    n2 = X_train.columns.values
    n2_Y = Y_train.columns.values
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_train = pd.DataFrame(X_train, columns=n2)

    clf = MLPRegressor(solver='adam', alpha=1e-5, hidden_layer_sizes=architecture, random_state=1)
    #clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1, 1), random_state=1)
    print("Fitting...", flush=True)
    clf.fit(X_train,Y_train)

    [X_test, Y_test] = CreateSet(test, interpolate, applyWindTransformation)

    X_test = scaler.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=n2)
    print("Predicting...", flush=True)
    predictions = clf.predict(X_test)

    predictions = pd.DataFrame(predictions, columns=n2_Y)
    #print(predictions)
    #print(Y_test)
    predictions.loc[:, predictions.columns.str.startswith('Wind m/s')] = predictions.loc[:, predictions.columns.str.startswith('Wind m/s')].applymap(WindToBool)
    Y_test.loc[:, Y_test.columns.str.startswith('Wind m/s')] = Y_test.loc[:, Y_test.columns.str.startswith('Wind m/s')].applymap(WindToBool)

    wind_predictions = predictions.loc[:, predictions.columns.str.startswith('Wind m/s')].any(axis='columns')
    wind_Y =  Y_test.loc[:, Y_test.columns.str.startswith('Wind m/s')].any(axis='columns')

    good_predictions = wind_predictions == wind_Y
    good_predictions = good_predictions[good_predictions == True].sum()

    errorsTemperature = abs(predictions.loc[:, predictions.columns.str.startswith('Temperature')]-Y_test.loc[:, Y_test.columns.str.startswith('Temperature')])
    np.savetxt('data'+ architecture + '-' + test + '-' + interpolate + applyWindTransformation +'.csv', errorsTemperature, delimiter=',')
    print("Saving...", flush=True)
    with open(timestr + '.csv', "a+") as myfile:
        myfile.write(name + ',' + train + ',' + test + ',' + str(architecture).replace(',', '-') + ',' + str(interpolate) + ',' + str(applyWindTransformation) + 
    ',' + str(np.average(errorsTemperature)) + ',' + str(np.std(errorsTemperature)) + ',' + str(good_predictions/(Y_test.shape[0]*Y_test.shape[1]/2)*100*8) + '\n')
    #print(name, ',', train, ',', test, ',', architecture, ',', interpolate, ',', applyWindTransformation, ',', )
    #print('Temperature\nAverage error: ' + str(np.average(errorsTemperature)) + 'Average std: ' + str(np.std(errorsTemperature).mean()))
    #print('Wind\nGood predictions: ' + str(good_predictions/(Y_test.shape[0]*Y_test.shape[1]/2)*100*8) + "%")

architectures = [(5, 10), (10, 10), (15, 10), (20, 10), (25, 10), (30, 10), (30, 5), (30, 10), (30, 15), (30, 20), (30, 25)]

train1 = "data/train_1"
test1 = "data/test_1"
train2 = "data/train_2"
test2 = "data/test_2"
with open(timestr + '.csv', "a+") as myfile:
    myfile.write('name,train,test,architecture,inteprolate,applyWindTransformation,temperatureAvgError,temperatureAvgStd,windGoodPredictions\n')
for architecture in architectures:
    CreateTestinScenario("Default test", train1, test1, architecture, 0, 0)
    CreateTestinScenario("Default test", train2, test2, architecture, 0, 0)

    CreateTestinScenario("Default test", train1, test1, architecture, 1, 0)
    CreateTestinScenario("Default test", train2, test2, architecture, 1, 0)

    CreateTestinScenario("Default test", train1, test1, architecture, 0, 1)
    CreateTestinScenario("Default test", train2, test2, architecture, 0, 1)

    CreateTestinScenario("Default test", train1, test1, architecture, 1, 1)
    CreateTestinScenario("Default test", train2, test2, architecture, 1, 1)