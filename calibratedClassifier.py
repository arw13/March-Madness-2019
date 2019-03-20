# Using the Calibrated Classifier from scikit, try to improve probabilisitc classification
import numpy as np
import pandas as pd
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split


df_features = pd.read_csv('./MarchMadnessFeatures.csv')
X = df_features.iloc[:,1:]
xDim = np.shape(X)[1]
X_train = X.values.reshape(-1,xDim)
y_train = df_features.Result.values

print('Feature vector dimension is: %.2f' % xDim)

df_test = pd.read_csv('./MarchMadnessTest.csv')

X_test = df_test.iloc[:,1:]
xDimTest = np.shape(X_test)[1]
X_test = X_test.values.reshape(-1,xDimTest)
y_test = df_test.Result.values

# Gaussian Naive-Bayes with no calibration
clf = GaussianNB()
clf.fit(X_train, y_train)  # GaussianNB itself does not support sample-weights
prob_pos_clf = clf.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes with isotonic calibration
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
clf_isotonic.fit(X_train, y_train)
prob_pos_isotonic = clf_isotonic.predict_proba(X_test)[:, 1]

# Gaussian Naive-Bayes with sigmoid calibration
clf_sigmoid = CalibratedClassifierCV(clf, cv=2, method='sigmoid')
clf_sigmoid.fit(X_train, y_train)
prob_pos_sigmoid = clf_sigmoid.predict_proba(X_test)[:, 1]

print("Brier scores: (the smaller the better)")

clf_score = brier_score_loss(y_test, prob_pos_clf)
print("No calibration: %1.3f" % clf_score)

clf_isotonic_score = brier_score_loss(y_test, prob_pos_isotonic)
print("With isotonic calibration: %1.3f" % clf_isotonic_score)

clf_sigmoid_score = brier_score_loss(y_test, prob_pos_sigmoid)
print("With sigmoid calibration: %1.3f" % clf_sigmoid_score)

'''-------------- Make prediction -------------------------'''
data_dir = './'
df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage1.csv')
data_file = data_dir + 'MarchMadnessAdvStats.csv'
df_adv = pd.read_csv(data_file)
df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')


n_test_games = len(df_sample_sub)

def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))

def seed_to_int(seed):
    '''Get just the digits from the seeding. Return as int'''
    s_int = int(seed[1:3])
    return s_int

print('Loading data for submission test')

# Make the seeding an integer
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(columns=['Seed'], inplace=True) # This is the string label
df_seeds.head()


T1_seed = []
T1_adv = []
T2_adv = []
T2_seed = []
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    t1_seed = df_seeds[(df_seeds.TeamID == t1) & (df_seeds.Season == year)].seed_int.values[0]
    t2_seed = df_seeds[(df_seeds.TeamID == t2) & (df_seeds.Season == year)].seed_int.values[0]
    t1_adv = df_adv[(df_adv.TeamID == t1) & (df_adv.Season == year)].values[0]
    t2_adv = df_adv[(df_adv.TeamID == t2) & (df_adv.Season == year)].values[0]
    T1_seed.append(t1_seed)
    T1_adv.append(t1_adv)
    T2_seed.append(t2_seed)
    T2_adv.append(t2_adv)

T1_adv = [row[2:] for row in T1_adv]
T2_adv = [row[2:] for row in T2_adv]
T1_seed = np.reshape(T1_seed, [n_test_games,-1]).tolist()
T2_seed = np.reshape(T2_seed, [n_test_games, -1]).tolist()
X_pred = np.concatenate((T1_seed, T1_adv, T2_seed, T2_adv), axis=1)

df_subData = pd.DataFrame(np.array(X_pred).reshape(np.shape(X_pred)[0], np.shape(X_pred)[1]))

xDim = np.shape(df_subData)[1]
X_pred = df_subData.values.reshape(-1,xDim)

preds = clf.predict_proba(X_pred)

# df_sample_sub = pd.DataFrame()
clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = preds
print("Submission shape",df_sample_sub.shape)

filename = 'GaussNB'
save_dir = './'
c=0
ext = '.csv'
if os.path.exists(save_dir+filename+ext):
    while os.path.exists(filename+ext):
        c+=1
    filename = filename+'_'+str(c)
    df_sample_sub.to_csv(save_dir+filename+ext, index=False)
else:
    df_sample_sub.to_csv(save_dir+filename+ext, index=False)
