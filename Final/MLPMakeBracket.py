import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras import regularizers, optimizers
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV


df_features = pd.read_csv('./Data/MarchMadnessFeatures.csv')
X = df_features.iloc[:,1:]
xDim = np.shape(X)[1]
X_train = X.values.reshape(-1,xDim)
y_train = df_features.Result.values

# Testing feature scaling
X_train = preprocessing.scale(X_train)

print('Feature vector dimension is: %.2f' % xDim)

'''------------------MLP-------------------------------'''
dropRate = 0.5
numBatch = 20
numEpoch = 150
learningRate = 1e-5

# MLP model
MLP = Sequential()
MLP.name = 'MLP'
MLP.add(Dense(3*xDim, input_dim=xDim, kernel_initializer='glorot_normal',activation = 'tanh'))
MLP.add(Dropout(dropRate))
MLP.add(Dense(200,kernel_initializer='glorot_normal',activation = 'tanh'))
MLP.add(Dropout(dropRate))
MLP.add(Dense(200, kernel_initializer='glorot_normal',activation = 'tanh'))
MLP.add(Dropout(dropRate))
MLP.add(Dense(1, kernel_initializer='glorot_normal', activation='sigmoid'))

# Compile model
adam = optimizers.Adam(lr=learningRate, decay=1e-6, amsgrad=True)
optim = optimizers.Adadelta()
MLP.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

MLP.fit(X_train, y_train, epochs=numEpoch, batch_size=numBatch)


# ## Making predictions with model

# ### Extract data desired

data_dir = './Data/'
df_sample_sub = pd.read_csv(data_dir + 'SampleSubmissionStage2.csv')
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


preds = MLP.predict(X_pred)

# df_sample_sub = pd.DataFrame()
# clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = preds
df_sample_sub.shape


filename = 'makeBrack'
save_dir = '../March-Madness/Final/'
c=0
ext = '.csv'
if os.path.exists(save_dir+filename+ext):
    while os.path.exists(filename+ext):
        c+=1
    filename = filename+'_'+str(c)
    df_sample_sub.to_csv(save_dir+filename+ext, index=False)
else:
    df_sample_sub.to_csv(save_dir+filename+ext, index=False)

