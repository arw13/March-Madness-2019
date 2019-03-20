# # Data Org Notebook

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import pi

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "./Data"]).decode("utf8"))


# ### Lock Random Seed

np.random.seed(13)


# # Load and Organize Training data
# loads the data from the dataset from csv as a pandas dataframe
# the output from pd.read_csv is a dataframe. dataframes have certain callable properties such as head(), which allows you to see the first five rows of the dataframe
data_dir = './Data/'
df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
df_tour = pd.read_csv(data_dir + 'NCAATourneyCompactResults.csv')
df_tour.head()



# Incorporate Massey Ordinals - MAS, SAG, POM
# the Massey ordinal dataset is a compilation of different ranking system. The ones I chose are the most reliable/most comprehensive
# First I select the end of season ranking and seasons 2008 and later
# Then I merge the dataframes together, and choose the ranking categories I want

#Load Data
df_rank = pd.read_csv('./Data/MasseyOrdinals.csv')
#Choose Latest Ranking 
df_rank = df_rank[df_rank.RankingDayNum>=133]
df_rank = df_rank[df_rank.Season>=2008]

#Selectr rankings of interest and make each ranking system ranking into a separate column
df_merge = pd.merge(df_rank.loc[df_rank['SystemName']=='MAS'],
                    df_rank.loc[df_rank['SystemName']=='SAG'], how='left', 
                    on=['Season', 'TeamID', 'RankingDayNum'])
df_rank = pd.merge(left=df_merge, right=df_rank.loc[df_rank['SystemName']=='POM'], 
                  how='left', on=['Season', 'TeamID', 'RankingDayNum'] )


df_rank.shape


# ### Remove unnecessary cols
# Just a little clean up to remove extra column labels

# Drop
df_rank.drop(labels=['SystemName_x', 'SystemName_y', 'SystemName','RankingDayNum'], inplace=True, axis =1)
df_rank.rename(columns={'OrdinalRank_x':'MAS', 'OrdinalRank_y':'SAG', 'OrdinalRank':'POM'}, inplace=True)

# df_rank.head()


# Add advanced stats from https://www.kaggle.com/lnatml/feature-engineering-with-advanced-stats/notebook
# 
# This uses some equations I found in the above kernel to create some advanced stats. Not sure how useful they are, since in theory a neural network should be able to approximate them as they are all linear relations but it saves some training if so to do this math now. 


df = pd.read_csv(data_dir+'RegularSeasonDetailedResults.csv')

#Points Winning/Losing Team
df['WPts'] = df.apply(lambda row: 2*(row.WFGM-row.WFGM3) + 3*row.WFGM3 + row.WFTM, axis=1)
df['LPts'] = df.apply(lambda row: 2*(row.LFGM-row.WFGM3) + 3*row.LFGM3 + row.LFTM, axis=1)

#Calculate Winning/losing Team Possesion Feature
wPos = df.apply(lambda row: 0.96*(row.WFGA + row.WTO + 0.44*row.WFTA - row.WOR), axis=1)
lPos = df.apply(lambda row: 0.96*(row.LFGA + row.LTO + 0.44*row.LFTA - row.LOR), axis=1)
#two teams use almost the same number of possessions in a game
#(plus/minus one or two - depending on how quarters end)
#so let's just take the average
df['Pos'] = (wPos+lPos)/2

#Offensive efficiency (OffRtg) = 100 x (Points / Possessions)
df['WOffRtg'] = df.apply(lambda row: 100 * (row.WPts / row.Pos), axis=1)
df['LOffRtg'] = df.apply(lambda row: 100 * (row.LPts / row.Pos), axis=1)
#Defensive efficiency (DefRtg) = 100 x (Opponent points / Opponent possessions)
df['WDefRtg'] = df.LOffRtg
df['LDefRtg'] = df.WOffRtg
#Net Rating = Off.eff - Def.eff
df['WNetRtg'] = df.apply(lambda row:(row.WOffRtg - row.LDefRtg), axis=1)
df['LNetRtg'] = df.apply(lambda row:(row.LOffRtg - row.LDefRtg), axis=1)

#Assist Ratio : Percentage of team possessions that end in assists
df['WAstR'] = df.apply(lambda row: 100 * row.WAst / (row.WFGA + 0.44*row.WFTA + row.WAst + row.WTO), axis=1)
df['LAstR'] = df.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)
#Turnover Ratio: Number of turnovers of a team per 100 possessions used.
#(TO * 100) / (FGA + (FTA * 0.44) + AST + TO
df['WTOR'] = df.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)
df['LTOR'] = df.apply(lambda row: 100 * row.LAst / (row.LFGA + 0.44*row.LFTA + row.LAst + row.LTO), axis=1)

#The Shooting Percentage : Measure of Shooting Efficiency (FGA/FGA3, FTA)
df['WTSP'] = df.apply(lambda row: 100 * row.WPts / (2 * (row.WFGA + 0.44 * row.WFTA)), axis=1)
df['LTSP'] = df.apply(lambda row: 100 * row.LPts / (2 * (row.LFGA + 0.44 * row.LFTA)), axis=1)
#eFG% : Effective Field Goal Percentage adjusting for the fact that 3pt shots are more valuable
df['WeFGP'] = df.apply(lambda row:(row.WFGM + 0.5 * row.WFGM3) / row.WFGA, axis=1)
df['LeFGP'] = df.apply(lambda row:(row.LFGM + 0.5 * row.LFGM3) / row.LFGA, axis=1)
#FTA Rate : How good a team is at drawing fouls.
df['WFTAR'] = df.apply(lambda row: row.WFTA / row.WFGA, axis=1)
df['LFTAR'] = df.apply(lambda row: row.LFTA / row.LFGA, axis=1)

#OREB% : Percentage of team offensive rebounds
df['WORP'] = df.apply(lambda row: row.WOR / (row.WOR + row.LDR), axis=1)
df['LORP'] = df.apply(lambda row: row.LOR / (row.LOR + row.WDR), axis=1)
#DREB% : Percentage of team defensive rebounds
df['WDRP'] = df.apply(lambda row: row.WDR / (row.WDR + row.LOR), axis=1)
df['LDRP'] = df.apply(lambda row: row.LDR / (row.LDR + row.WOR), axis=1)
#REB% : Percentage of team total rebounds
df['WRP'] = df.apply(lambda row: (row.WDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1)
df['LRP'] = df.apply(lambda row: (row.LDR + row.WOR) / (row.WDR + row.WOR + row.LDR + row.LOR), axis=1)


# After creating adv stats, now we need to split the winning and losing team stats for a more overall picture

# In[ ]:


df_TeamID = pd.concat((df['WTeamID'], df['LTeamID']), axis=1)
df_adv2 = df.iloc[:, 34:].copy()
df_adv = pd.concat((df_TeamID, df_adv2), axis=1)
df_adv = pd.concat((df['Season'], df_adv), axis=1)
# df_adv.head()
names = df_adv.columns.values
print(names)
df_adv.shape


Wnames = ['Season', 'WTeamID','WPts','Pos', 'WOffRtg' ,'WDefRtg',
          'WNetRtg', 'WAstR', 'WTOR', 'WTSP','WeFGP','WFTAR', 'WORP', 'WDRP','WRP']
Lnames = ['Season', 'LTeamID', 'LPts', 'Pos', 'LOffRtg','LDefRtg',
          'LNetRtg', 'LAstR', 'LTOR','LTSP', 'LeFGP','LFTAR', 'LORP', 'LDRP', 'LRP' ]
df_advW = df_adv.loc[:,Wnames].copy()
df_advL = df_adv.loc[:,Lnames].copy()
df_advW.rename(columns={'WTeamID':'TeamID'}, inplace=True)
df_advL.rename(columns={'LTeamID':'TeamID'}, inplace=True)



df_advL.shape


# Must concat then group the advanced stats to get season averages for each team

names = ['Season', 'TeamID', 'Pts', 'Pos', 'OffRtg','DefRtg',
          'NetRtg', 'AstR', 'TOR','TSP', 'eFGP','FTAR', 'ORP', 'DRP', 'RP' ]
df_advL.columns = names
df_advW.columns = names
df_A = pd.concat((df_advL, df_advW), axis=0, ignore_index=True)
groupedA = df_A.groupby(['Season', 'TeamID'], as_index=False)
df_A = groupedA.agg(np.mean)
# df_A.shape
df_advL = df_A.copy()
df_advW = df_A.copy()


df_Adv= pd.merge(left=df_A, right=df_rank, on=['Season', 'TeamID'])
df_Adv.tail()


# Cut seasons prior to Max Ranking date

df_seeds = df_seeds[df_seeds.Season>=min(df_rank.Season) ]
df_seeds = df_seeds[ df_seeds.Season<=max(df_rank.Season)]
df_tour = df_tour[df_tour.Season>=min(df_rank.Season)]
df_tour = df_tour[ df_tour.Season<=max(df_rank.Season)]
# df_tour.head()

df_tour.shape

# Cut off the region identifier from the seed number

def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(columns=['Seed'], inplace=True) # This is the string label

df_tour.drop(labels=['DayNum', 'WScore', 'LScore','WLoc', 'NumOT'], inplace=True, axis=1)
# df_tour.head()


# Merge the Seeds with their corresponding TeamIDs in the compact results dataframe.

df_winSeeds = df_seeds.rename(columns={'TeamID':'WTeamID', 'seed_int':'WSeed'})
df_lossSeeds = df_seeds.rename(columns={'TeamID':'LTeamID', 'seed_int':'LSeed'})


df_dummy = pd.merge(left=df_tour, right=df_lossSeeds, how='left', on=['Season', 'LTeamID'])

df_concat = pd.merge(left=df_dummy, right=df_winSeeds, how='left' ,on=['Season', 'WTeamID'])
# df_concat['SeedDiff'] = df_concat.WSeed - df_concat.LSeed
# df_concat.head()
df_concat.shape


#  Make a winner and loser dataframe with seed, relative seed, and win/loss. 

df_wins = pd.DataFrame()
df_wins['Seed'] = df_concat['WSeed']
# df_wins['SeedDiff'] = df_concat['SeedDiff']
df_wins['TeamID'] = df_concat['WTeamID']
df_wins['Season'] = df_concat['Season']
df_wins['Result'] = 1
df_wins = pd.merge(left=df_wins, right=df_advW, how='left', on=['Season', 'TeamID'])
df_wins = pd.merge(left=df_wins, right=df_rank, how='left', on=['Season', 'TeamID'])
# df_wins.tail()


df_losses = pd.DataFrame()
df_losses['Seed'] = df_concat['LSeed']
# df_losses['SeedDiff'] = -df_concat['SeedDiff']
df_losses['TeamID'] = df_concat['LTeamID']
df_losses['Season'] = df_concat['Season']
df_losses['Result'] = 0
df_losses = pd.merge(left=df_losses, right=df_advL, how='left', on=['Season', 'TeamID'])
df_losses = pd.merge(left=df_losses, right=df_rank, how='left',on=['Season', 'TeamID'])
# df_losses.tail()


# Prefix the cols with 'Opp' in dummy frames to allow for horizontal concat in order to expand data to be
# | winr_stat    losr_stat |
# | losr_stat    winr_stat |

df_lossesOpp = df_losses.copy()

df_lossesOpp.drop(labels=['Season', 'Result'], inplace=True, axis=1)
new_names = [(i,'Opp'+i) for i in df_lossesOpp.columns.values]
df_lossesOpp.rename(columns = dict(new_names), inplace=True)
# df_lossesOpp.tail()

df_winsOpp = df_wins.copy()

df_winsOpp.drop(labels=['Season', 'Result'], inplace=True, axis=1)
new_names = [(i,'Opp'+i) for i in df_winsOpp.columns.values]
df_winsOpp.rename(columns = dict(new_names), inplace=True)
# df_winsOpp.head()


df_winloss = pd.concat([df_wins, df_lossesOpp], axis=1)
df_losswin = pd.concat([df_losses, df_winsOpp], axis=1)


# # Combine into final dataframe
# This will be the input to the neural network 

df_finalData = pd.concat((df_winloss, df_losswin))
results = df_finalData['Result']
df_finalData.drop(labels=['Result'], inplace=True, axis=1)
df_finalData.insert(0, 'Result', results)
df_finalData.shape

 ## Create final dataframe
# Remove team ID and season -> cannot be scaled and not a metric. 
# Including a large integer identifier is not helpful in a neural network. A NN works to combine all the inputs, meaning a big ol number will skew the outputs, even if it is just an ID number


# df_finalData = df_finalData[df_finalData<2014]
df_finalData.drop(labels=['TeamID', 'Season', 'OppTeamID'], inplace=True, axis=1)
# df_finalData.head()

# Check for null values (NaNs etc) in case I messed up in any of the math or merges. If True, then there is a null value in one of those columns

df_finalData.isnull().any()

# ###  Save dataframes as csv

df_finalData.to_csv('MarchMadnessFeatures_allSeasons.csv', index=False)


df_Adv.to_csv('MarchMadnessAdvStats_allSeasons.csv', index=False)

