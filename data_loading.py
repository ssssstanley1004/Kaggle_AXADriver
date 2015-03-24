
import numpy as np
import pandas as pd
from multiprocessing import Pool
import datetime
import itertools
import os


def data_loading(mainDir,n_quantile):
    # @Param mainDir: A string variable for main directory where all dirver trips are stored in a "drivers" folder.
    # @Return a pd.DataFrame for all features
    
    features=pd.DataFrame()
    os.chdir(mainDir)
    all_drivers = [s for s in os.listdir('.') if os.path.isdir(s)]
    
    pool=Pool(4)
    for i, feature in enumerate(pool.imap(load_single_driver_tuple,itertools.izip(itertools.repeat(mainDir),all_drivers,itertools.repeat(n_quantile)))):
        features = features.append(feature)
        print 'loading %dth driver finished at %s: ' % (i, datetime.datetime.now())
    
    return features
    
    
    
def load_single_driver(mainDir,driver_id, n_quantile=20):
    
    # @Param driver_id: load trips for current driver. 
    # @Param n_quantile: A number for quantile variable to get quantile features.
    # @Return pd.Dataframe calculated features for all trips. 
    
    feature=pd.DataFrame()
    for i, trip_id in enumerate([file for file in os.listdir(mainDir + '\\%s' % driver_id) if file.endswith('.csv') or file.endswith('.CSV') ]):
        tripDir = mainDir +'\\%s\\%s' %(driver_id,trip_id)
        trip = pd.read_csv(tripDir,header=False,sep=',')
        ifeature = get_trip_feature(trip,driver_id,trip_id[:-4],n_quantile)
        feature = feature.append(ifeature)
    feature['driver_id']=int(driver_id)
    
    return feature
        

def load_single_driver_tuple(params):
    return load_single_driver(params[0],params[1],params[2])

        

def get_trip_feature(trip,driver_id,trip_id,n_quantile=20):
    
    # @Param trip: a DataFrame for all locations
    tripDiff=trip.diff(10)
    # Take average of 10 locations
    speed = np.sqrt((tripDiff**2).sum(axis=1))
    acceleration = speed.diff()
    jerk = acceleration.diff()
    direction = np.arctan2(tripDiff['x'],tripDiff['y']).diff()

    # get quantile features
    ispeed, name_ispeed = get_quantile(speed,n_quantile,'speed')
    iacceleration, name_accel = get_quantile(acceleration,n_quantile,'acceleration')
    idirection, name_direction = get_quantile(direction,n_quantile,'direction')
    ijerk, name_jerk = get_quantile(jerk,n_quantile,'jerk')
    
    # get volatility score by acceleration and jerk with reduced quantiles.
    iaccScore,name_accScore = get_volatility_score(acceleration,n_quantile/2,'acceleration_score')
    ijerkScore,name_jerkScore = get_volatility_score(jerk,n_quantile/2,'jerk_score')
  
    ifeature=pd.DataFrame(np.vstack([ispeed+iacceleration+idirection+ijerk+iaccScore+ijerkScore]),
                        columns = name_ispeed+name_accel+name_direction+name_jerk+name_accScore+name_jerkScore)
    ifeature['trip_id']=int(trip_id)
    return ifeature
    
def get_quantile(feature,n_quantile,name='speed'):
    
    ispeed,iname =[],[]
    for i in np.linspace(0,1,n_quantile):
        qspeed = feature.quantile(i)
        qname = name+'_q_%.2f' % (i)
        ispeed.append(qspeed)
        iname.append(qname)
    return ispeed,iname


def get_volatility_score(feature,n_quantile=10, name='acceleration'):

    """ Compute volatility score = # of volatile / total # """

    score, name_score= [],[]
    group_index = pd.cut(feature.index,n_quantile,labels=False)    
    
    for i in range(0,n_quantile):
        qfeature=feature[group_index==i]
        mean = np.mean(qfeature)
        std = np.std(qfeature)
        qscore = sum([(qfeature.iloc[j] >(mean+std) or qfeature.iloc[j]<(mean-std)) for j in range(0,len(qfeature))]) / float(len(qfeature))
        score.append(qscore)
        name_score.append(name+'_q_%d'%(i))
    return score,name_score
    

if __name__=='__main__':
    
    n_quantile = 20
    main_dir='..\\kaggle\\AXADrivers\\drivers'
    
    features = data_loading(main_dir,n_quantile)
    features.to_csv(main_dir + '\\features.csv', sep=',', header=True, index=False)
