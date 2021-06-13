import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from ast import literal_eval
from apyori import apriori


def Neighbour(df_learner,input1,neighbour=10):
    y = df_learner['Learner ID_encoded']
    X = df_learner.drop(['Learner ID', 'Learner ID_encoded'], axis=1)
    MMS = MinMaxScaler()
    X_scaled = MMS.fit_transform(X)

    def LeanerPredict(database, NewLearner, MMS):
        NewLearner = MMS.transform([NewLearner])
        df = pd.DataFrame()
        i = 1
        for data in database:
            dist = np.linalg.norm(data - NewLearner)
            dicti = {
                'Learner': 'Learner' + str(i),
                'dist': dist
            }
            df = df.append(dicti, ignore_index=True)
            i = i + 1
        return df

    distance = LeanerPredict(X_scaled, input1, MMS)
    distance = distance.sort_values(by='dist')
    top_n = distance.head(neighbour)
    return top_n

def preprocessing(df,unit,neighbours):
    a=[]
    for i in range(neighbours):
        a1=literal_eval(df[unit].values[i])
        #print(a1)
        #print(i)
        a.append(a1)
    return a


def apri(x):
    results = list(apriori(x, max_length=5, min_confidence=0.4, min_support=0.3, min_lift=1))
    df = pd.DataFrame()
    for result in results:
        dicti = {
            'items': result.items,
            'Support': result.support,
            'len': len(result.items)
        }
        df = df.append(dicti, ignore_index=True)
    df = df.sort_values(by='Support', ascending=False)
    maxlen = df['len'].max()
    # print(maxlen)
    a = list(df[df['len'] == maxlen]['items'].values[0])
    # print(a)
    if maxlen > 1:
        b = list(df[df['Support'] > 0.3]['items'].values[0])
        # print(b)
        c = np.union1d(a, b)
        return c
    else:
        return a
