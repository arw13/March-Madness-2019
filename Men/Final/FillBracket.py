
# coding: utf-8

# In[9]:


## Script to match teams to names to use to make a bracket

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from math import pi

from subprocess import check_output
print(check_output(["ls", "../Final"]).decode("utf8"))


# In[43]:


# df_pred = pd.read_csv(data_dir + 'MLP_noclipping_0.csv')
df_pred = pd.read_csv('./MLP_bracket.csv')
df_Teams = pd.read_csv('./Data/Teams.csv')


# In[44]:


df_Teams.head()


# In[45]:


ID = []
Team = []
for ii, row in df_Teams.iterrows():
    ID.append(row.TeamID)
    Team.append(row.TeamName)
name_dict = dict(zip(ID, Team))


# In[46]:


def get_year_t1_t2(ID):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in ID.split('_'))


# In[47]:


Team1 = []
Team2 =[]
for ii, row in df_pred.iterrows():
    year, t1, t2 = get_year_t1_t2(row.ID)
    Team1.append(name_dict[t1])
    Team2.append(name_dict[t2])
    


# In[48]:


Team1 = np.asarray(Team1)
Team2 = np.asarray(Team2)

df_pred['Team1'] = Team1
df_pred['Team2'] = Team2


# In[49]:


df_pred.to_csv('./preds_w_Names.csv', index=False)

