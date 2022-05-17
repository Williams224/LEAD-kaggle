from venv import create
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import RocCurveDisplay
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

def create_submission(clf, test_set_features, f):
    X = test_set_features[f]
    print(X.info())
    preds = clf.predict(X.values)
    df_submit = test_set_features[['row_id']]
    df_submit['anomaly'] = preds
    return df_submit


if __name__ == "__main__":

    df = pd.read_parquet( "/Users/TimothyW/Fun/energy-kaggle/data/prepped_train_features.parquet")
    
    features = ['meter_reading','air_temperature','cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed','hour','weekday','24h_avg','24h_std','z_score']
    
    X = df[features]
    y = df[['anomaly']]

    group_kf = GroupKFold(n_splits = 3)
    groups = df.building_id.values

    clfs = []

    for train_index, test_index in group_kf.split(X,y, groups=groups):    
        print(train_index)    
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        train_data = Pool(data=X_train,
                  label=y_train)

        test_data = Pool(data = X_test, label = y_test)

        clf = CatBoostClassifier(iterations=100)

        clf.fit(train_data, eval_set = test_data, verbose = True)

        clfs.append(clf)

        predictions = clf.predict(X_test)

        display = RocCurveDisplay.from_predictions(y_test, predictions)

        #display.plot()
        #plt.show()

        print(roc_auc_score(y_test, predictions))

    #df_test = pd.read_parquet('prepped_test_features.parquet')

    #df_submit = create_submission(clfs[1],df_test, features)    



    #check discrete groups + functions



        

    



    