# Kaggle_AXADriver
Code for Kaggle AXA Driver Telematics Analysis competition. 


1. Data Loading
  Load all driver location data and compute feature. Features used in the model are basic quantile features.
    1. speed
    2. acceleration
    3. direction. driving angle
    4. jerk. Second derivative of speed. Changes of acceleration
    5. Acceleration volatility score. The percentage of acceleration outliers
    6. Jerk volatility score. The percentage of jerk outliers


2. Prediction (model.py)
  Pick trips of current driver as postive trips and randomly select negative trip from trips of other drivers.  RandomForestClassifer with KFold cross-validation is applied. 

3. Learnings and to be added
  Learnings from high scoring winners' code is a invaluable experience. 
    1. Trip matching ⑴: 
    2. Adding more features: Volecity and centripetal acceleration can be used in this competition ⑵. 

Reference:
⑴http://www.kaggle.com/c/axa-driver-telematics-analysis/forums/t/12850/trip-matching-our-methods
⑵https://github.com/alzmcr/axa
