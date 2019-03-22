'''Load in sample data and order for input into MLP'''
import pandas as pd
import numpy as np
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

