###  Kaggle_AXADriver
Code for Kaggle AXA Driver Telematics Analysis competition.


### Data Loading:
Load all driver location data and compute feature. Features used in the model are basic quantile features.

* speed
* acceleration
* direction. driving angle
* jerk. Second derivative of speed. Changes of acceleration
* Acceleration volatility score. The percentage of acceleration outliers
* Jerk volatility score. The percentage of jerk outliers


### Prediction (model.py)
Pick trips of current driver as postive trips and randomly select negative trip from trips of other drivers.  RandomForestClassifer with KFold cross-validation is applied.

### Learnings and to be added

* Trip matching ⑴: 
* Adding more features⑵.   

### Reference

* http://www.kaggle.com/c/axa-driver-telematics-analysis/forums/t/12850/trip-matching-our-methods
* http://www.kaggle.com/c/axa-driver-telematics-analysis/forums/t/12849/github-repos-now-live

