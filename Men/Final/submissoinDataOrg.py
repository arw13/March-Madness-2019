
# coding: utf-8

# # Data Organization Script # 

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import pi
# import seaborn as sns
# import matplotlib as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# Any results you write to the current directory are saved as output.


# # Load and Organize Training data
# 

# In[2]:


data_dir = './Data/'
df_seeds = pd.read_csv(data_dir + 'NCAATourneySeeds.csv')
# sdf_tour.head()


# Incorporate Massey Ordinals - MAS, SAG, POM

# In[3]:


#Load Data
df_rank = pd.read_csv(data_dir+ 'MasseyOrdinals_2019.csv')
#Choose Latest Ranking 
df_rank = df_rank[df_rank.RankingDayNum>=133]
df_rank = df_rank[df_rank.Season==2019]

#Selectr rankings of interest and make each ranking system ranking into a separate column
df_merge = pd.merge(df_rank.loc[df_rank['SystemName']=='MAS'],
                    df_rank.loc[df_rank['SystemName']=='SAG'], how='left', 
                    on=['Season', 'TeamID', 'RankingDayNum'])
df_rank = pd.merge(left=df_merge, right=df_rank.loc[df_rank['SystemName']=='POM'], 
                  how='left', on=['Season', 'TeamID', 'RankingDayNum'] )


# Remove unnecessary cols

# In[4]:


# Drop
df_rank.drop(labels=['SystemName_x', 'SystemName_y', 'SystemName','RankingDayNum'], inplace=True, axis =1)
df_rank.rename(columns={'OrdinalRank_x':'MAS', 'OrdinalRank_y':'SAG', 'OrdinalRank':'POM'}, inplace=True)

# df_rank.head()


# Add advanced stats from https://www.kaggle.com/lnatml/feature-engineering-with-advanced-stats/notebook

# In[5]:


df = pd.read_csv(data_dir+'RegularSeasonDetailedResults.csv')
df = df[df.Season==2019]


# In[6]:


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

# In[7]:


df_TeamID = pd.concat((df['WTeamID'], df['LTeamID']), axis=1)
df_adv2 = df.iloc[:, 34:].copy()
df_adv = pd.concat((df_TeamID, df_adv2), axis=1)
df_adv = pd.concat((df['Season'], df_adv), axis=1)
# df_adv.head()
names = df_adv.columns.values 
print(names)


# In[8]:


Wnames = ['Season', 'WTeamID','WPts','Pos', 'WOffRtg' ,'WDefRtg',
          'WNetRtg', 'WAstR', 'WTOR', 'WTSP','WeFGP','WFTAR', 'WORP', 'WDRP','WRP']
Lnames = ['Season', 'LTeamID', 'LPts', 'Pos', 'LOffRtg','LDefRtg',
          'LNetRtg', 'LAstR', 'LTOR','LTSP', 'LeFGP','LFTAR', 'LORP', 'LDRP', 'LRP' ]
df_advW = df_adv.loc[:,Wnames].copy()
df_advL = df_adv.loc[:,Lnames].copy()
df_advW.rename(columns={'WTeamID':'TeamID'}, inplace=True)
df_advL.rename(columns={'LTeamID':'TeamID'}, inplace=True)


# Must concat then group the advanced stats to get season averages for each team

# In[9]:


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


# In[10]:


df_Adv= pd.merge(left=df_A, right=df_rank, on=['Season', 'TeamID'])
df_Adv.tail()


# Cut off the region identifier from the seed number

# In[11]:


def seed_to_int(seed):
    #Get just the digits from the seeding. Return as int
    s_int = int(seed[1:3])
    return s_int
df_seeds['seed_int'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(columns=['Seed'], inplace=True) # This is the string label


# In[13]:


df_Adv.to_csv(data_dir+'SubmissionMarchMadnessAdvStats.csv', index=False)

