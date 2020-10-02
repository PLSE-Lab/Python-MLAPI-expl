from __future__ import print_function
import numpy as np
import datetime
import csv
from sklearn import metrics
from sklearn.utils import shuffle
from sklearn import ensemble
from sklearn import cross_validation

species_map = {'CULEX RESTUANS' : "100000",
              'CULEX TERRITANS' : "010000", 
              'CULEX PIPIENS'   : "001000", 
              'CULEX PIPIENS/RESTUANS' : "101000", 
              'CULEX ERRATICUS' : "000100", 
              'CULEX SALINARIUS': "000010", 
              'CULEX TARSALIS' :  "000001",
              'UNSPECIFIED CULEX': "001100"} # hack!
def date(text):
    return datetime.datetime.strptime(text, "%Y-%m-%d").date()
    
def precip(text):
    TRACE = 1e-3
    text = text.strip()
    if text == "M":
        return None
    if text == "-":
        return None
    if text == "T":
        return TRACE
    return float(text)
    
def ll(text):
     return int(float(text)*100)/100

def impute_missing_weather_station_values(weather):
    # Stupid simple
    for k, v in weather.items():
        if v[0] is None:
            v[0] = v[1]
        elif v[1] is None:
            v[1] = v[0]
        for k1 in v[0]:
            if v[0][k1] is None:
                v[0][k1] = v[1][k1]
        for k1 in v[1]:
            if v[1][k1] is None:
                v[1][k1] = v[0][k1]
    
def load_weather():
    weather = {}
    for line in csv.DictReader(open("../input/weather.csv")):
        for name, converter in {"Date" : date,
                                "Tmax" : float,"Tmin" : float,"Tavg" : float,
                                "DewPoint" : float, "WetBulb" : float,
                                "PrecipTotal" : precip,"Sunrise" : precip,"Sunset" : precip,
                                "Depart" : float, "Heat" : precip,"Cool" : precip,
                                "ResultSpeed" : float,"ResultDir" : float,"AvgSpeed" : float,
                                "StnPressure" : float, "SeaLevel" : float}.items():
            x = line[name].strip()
            line[name] = converter(x) if (x != "M") else None
        station = int(line["Station"]) - 1
        assert station in [0,1]
        dt = line["Date"]
        if dt not in weather:
            weather[dt] = [None, None]
        assert weather[dt][station] is None, "duplicate weather reading {0}:{1}".format(dt, station)
        weather[dt][station] = line
    impute_missing_weather_station_values(weather)        
    return weather
    
    
def load_training():
    training = []
    for line in csv.DictReader(open("../input/train.csv")):
        for name, converter in {"Date" : date, 
                                "Latitude" : ll, "Longitude" : ll,
                                "NumMosquitos" : int, "WnvPresent" : int}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training
    
def load_testing():
    training = []
    for line in csv.DictReader(open("../input/test.csv")):
        for name, converter in {"Date" : date, 
                                "Latitude" : ll, "Longitude" : ll}.items():
            line[name] = converter(line[name])
        training.append(line)
    return training
    
    
def closest_station(lat, longi):
    # Chicago is small enough that we can treat coordinates as rectangular.
    stations = np.array([[41.995, -87.933],
                         [41.786, -87.752]])
    loc = np.array([lat, longi])
    deltas = stations - loc[None, :]
    dist2 = (deltas**2).sum(1)
    return np.argmin(dist2)
       
def normalize(X, mean=None, std=None):
    count = X.shape[1]
    if mean is None:
        mean = np.nanmean(X, axis=0)
    for i in range(count):
        X[np.isnan(X[:,i]), i] = mean[i]
    if std is None:
        std = np.std(X, axis=0)
    for i in range(count):
        X[:,i] = (X[:,i] - mean[i]) / std[i]
    return mean, std
    
def scaled_count(record):
    SCALE = 10.0
    if "NumMosquitos" not in record:
        # This is test data
        return 1
    return int(np.ceil(record["NumMosquitos"] / SCALE))
    
    
def assemble_X(base, weather):
    X = []
    for b in base:
        date = b["Date"]
        lat, longi = b["Latitude"], b["Longitude"]
        case = [date.year, date.month, date.day,date.weekday(), lat, longi]
        # Look at a selection of past weather values
        for days_ago in [1,2,3,5,8,12]:
            day = date - datetime.timedelta(days=days_ago)
            for obs in ["Tmax","Tmin","Tavg","DewPoint","WetBulb","PrecipTotal","Depart","Sunrise","Sunset","Heat","Cool","ResultSpeed","ResultDir"]:
                station = closest_station(lat, longi)
                case.append(weather[day][station][obs])
        # Specify which mosquitos are present
        species_vector = [float(x) for x in species_map[b["Species"]]]
        case.extend(species_vector)
        # Weight each observation by the number of mosquitos seen. Test data
        # Doesn't have this column, so in that case use 1. This accidentally
        # Takes into account multiple entries that result from >50 mosquitos
        # on one day. 
        for repeat in range(scaled_count(b)):
            X.append(case)    
    X = np.asarray(X, dtype=np.float32)
    return X
    
def assemble_y(base):
    y = []
    for b in base:
        present = b["WnvPresent"]
        for repeat in range(scaled_count(b)):
            y.append(present)    
    return np.asarray(y, dtype=np.int32).reshape(-1,1)


class AdjustVariable(object):
    def __init__(self, variable, target, half_life=20):
        self.variable = variable
        self.target = target
        self.half_life = half_life
    def __call__(self, nn, train_history):
        delta = self.variable.get_value() - self.target
        delta /= 2**(1.0/self.half_life)
        self.variable.set_value(np.float32(self.target + delta))

def cross_validate(estimator, X_train, y_train):
    """
    StratifiedKFold cross validation.
    """

    # cross validation
    cv = cross_validation.StratifiedKFold(y_train, n_folds=3, shuffle=True)
    scores = cross_validation.cross_val_score(estimator, X_train, y_train, scoring='roc_auc', cv=cv, n_jobs=-1)
    print('Cross Validation - scores:', ', '.join(str(round(score, 3)) for score in scores))
    print('Average score:', round(scores.mean(), 3))
    
    # Fit the model
    estimator
    estimator.fit(X_train, y_train)
    return estimator, scores
    
def train(classifier):
    weather = load_weather()
    training = load_training()
    
    X = assemble_X(training, weather)
    y = assemble_y(training).ravel()
    
    X, y = shuffle(X, y)

    classifier, scores = cross_validate(classifier, X, y)
    #classifier.fit(X, y)
    
    return classifier     
    

def submit(estimator):
    weather = load_weather()
    testing = load_testing()
    X = assemble_X(testing, weather) 
    predictions = estimator.predict_proba(X)[:, 1]    
    #
    out = csv.writer(open("submission_final_opt_extrarandomtrees.csv", "w"))
    out.writerow(["Id","WnvPresent"])
    for row, p in zip(testing, predictions):
        out.writerow([row["Id"], p])


if __name__ == "__main__":
    forest = ensemble.ExtraTreesClassifier(n_estimators=5000, class_weight='auto')
    forest = train(forest)
    submit(forest)
