
from sklearn import ensemble,cross_validation
import numpy as np
import pandas as pd
from multiprocessing import Pool
import datetime
import itertools



def classify_trip(tFeatures,driver_id):
    """ main function to make predictions """
    
    posFea = tFeatures[tFeatures['driver_id']==driver_id]
    
    # Sampling negative features
    tFeatures.index = np.arange(0, len(tFeatures))  # Reindexing before sampling
    n_negatives=2000
    negFea = tFeatures[tFeatures['driver_id']!=driver_id].loc[np.random.choice(tFeatures[tFeatures['driver_id']!=driver_id].index, n_negatives, replace=False)]
    trainFea = posFea.append(negFea) # combine into training set
    trainFea.index = np.arange(0,len(trainFea))
    
    X = trainFea;
    X['flag'] = X['driver_id']==driver_id
    X = X.set_index(['driver_id','trip_id']); y = X['flag']; X=X.drop('flag',1)
    pred = y.copy()
    
    # Training with kfold...
    RFC = ensemble.RandomForestClassifier(n_estimators = 500,max_depth=10, min_samples_leaf=3)
    
    print 'start predict driver: %d at %s' %(driver_id, datetime.datetime.now())
    for tr,cv in cross_validation.KFold(len(X),10,shuffle = True):
        X_tr, X_cv = X.iloc[tr], X.iloc[cv]
        y_tr, y_cv = y.iloc[tr], y.iloc[cv]
        if y_cv.sum()!=0: # if cv set is all false (no target driver trips are included in the cv set), skip the model prediction
            model = RFC.fit(X_tr, y_tr)
            pred[y_cv[y_cv].index.values]=model.predict_proba(X_cv[y_cv.values])[:,1]
    print 'prediction finished for driver: %d at %s' %(driver_id, datetime.datetime.now())
    
    return pred[y.values].reset_index()
    
    
def classify_trip_tuple(param):
    """ function with tuple of params instead of multiple arguments for pool.imap """
    return classify_trip(param[0],param[1])
    

if __name__=='__main__':
    
    data_file = 'C:\\Stanley\\python\\kaggle\\AXADrivers\\DEV\\driver_all\\driver_all.csv'
    submission_dir = 'C:\\Stanley\\python\\kaggle\\AXADrivers\\result'
    tFeatures = pd.read_csv(data_file,header=False,sep=',')
    tPred = pd.DataFrame()
    pool=Pool(4)
    driver_list = tFeatures['driver_id'].unique()[:]
    #driver_list = [1,2,3,4,100,200,300,400]
    for i,pred in enumerate(pool.imap(classify_trip_tuple,itertools.izip(itertools.repeat(tFeatures), driver_list))):
            tPred=tPred.append(pred)
    
    #for i in tFeatures['driver_id'].unique():
    #    pred = classify_trip(tFeatures,i)
    #    tPred = tPred.append(pred)
    
    print tPred[:].iloc[0:3]
    tPred['driver_trip']=tPred.apply(lambda x: '%d_%d' %(x['driver_id'],x['trip_id']), 1)
    submission = tPred[['driver_trip','flag']]; submission.columns = ['driver_trip','prob']
    submission.to_csv(submission_dir + '\\submission1.csv', header=True, index=False, sep=',')
    
    
    
    
