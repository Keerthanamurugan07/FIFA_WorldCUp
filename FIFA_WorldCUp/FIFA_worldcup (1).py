#!/usr/bin/env python
# coding: utf-8

# # FIFA World Cup History:

# ## 1.Top Performers :
# 

# ### a.which countries have won the most World Cups 
# 
#       

# #### Dataset : ' Worldcups'

# In[13]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

worldcups = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCups.csv")

top_winners = worldcups['Winner'].value_counts()

top_winners


# In[8]:


plt.figure(figsize=(10, 6))
top_winners.plot(kind='bar', color='skyblue')
plt.title('Countries with Most World Cup Wins')
plt.xlabel('Country')
plt.ylabel('Number of Wins')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# ### b. What are the top three most memorable finals in worldcup history

# #### Dataset :' WorldCupMatch'

# In[4]:


matches = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupMatches.csv")


# In[5]:


final_matches = matches[matches['Stage'] == 'Final']


# In[6]:


final_matches_summary = final_matches[['Year', 'Home Team Name', 'Home Team Goals', 'Away Team Goals', 'Away Team Name', 'Attendance', 'Win conditions']]


# In[7]:


final_matches_sorted = final_matches_summary.sort_values(by='Attendance', ascending=False)
memorable_finals = final_matches_sorted.head(3)


for index, row in memorable_finals.iterrows():
    print(f"### {int(row['Year'])} Final: {row['Home Team Name']} vs. {row['Away Team Name']}")
    print(f"- **Attendance:** {int(row['Attendance'])}")
    if pd.notna(row['Win conditions']):
        print(f"- **Win Condition:** {row['Win conditions']}")
    else:
        print(f"- **Win Condition:** N/A (Specific memorable event details can be added)")
    print("")


# ## 2. Host Country influence

# ### a. How often has the host country won (or) reached the finals

# #### Dataset : 'WorldCups'

# In[8]:


worldcups = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCups.csv")

host_wins = 0
host_finals = 0

for index, row in worldcups.iterrows():
    host_country = row['Country']
    winner = row['Winner']
    runner_up = row['Runners-Up']
    
    
    if host_country == winner:
        host_wins += 1
    
    
    if host_country == winner or host_country == runner_up:
        host_finals += 1


print(f"Host countries have won the World Cup {host_wins} times.")
print(f"Host countries have reached the finals {host_finals} times.")


# ###  b.Does the host country generally perform better than expected

# ##### Dataset : 'WorldCups' and 'WorldCupMatches'

# In[11]:


worldcups = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCups.csv")
matches = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupMatches.csv")


worldcups = worldcups.replace({'Quarter-finals': 'Quarter-Finals', 'Semi-finals': 'Semi-Finals'})
matches = matches.replace({'Quarter-finals': 'Quarter-Finals', 'Semi-finals': 'Semi-Finals'})


stages = ['Group Stage', 'Round of 16', 'Quarter-Finals', 'Semi-Finals', 'Third Place', 'Final', 'Winner']


stage_dict = {stage: i for i, stage in enumerate(stages)}


def get_stage(country, year):
    winner = worldcups.loc[worldcups['Year'] == year, 'Winner']
    runner_up = worldcups.loc[worldcups['Year'] == year, 'Runners-Up']
    third = worldcups.loc[worldcups['Year'] == year, 'Third']
    fourth = worldcups.loc[worldcups['Year'] == year, 'Fourth']
    
    if not winner.empty and country == winner.values[0]:
        return 'Winner'
    elif not runner_up.empty and country == runner_up.values[0]:
        return 'Final'
    elif not third.empty and country == third.values[0]:
        return 'Third Place'
    elif not fourth.empty and country == fourth.values[0]:
        return 'Semi-Finals'
    else:
        max_stage = matches[(matches['Year'] == year) & ((matches['Home Team Name'] == country) | (matches['Away Team Name'] == country))]['Stage'].max()
        return max_stage if pd.notna(max_stage) and max_stage in stage_dict else 'Group Stage'


host_performance = []
for index, row in worldcups.iterrows():
    host_country = row['Country']
    year = row['Year']
    stage = get_stage(host_country, year)
    host_performance.append(stage_dict[stage])


average_host_performance = sum(host_performance) / len(host_performance)


overall_performance = []
for index, row in matches.iterrows():
    home_country = row['Home Team Name']
    away_country = row['Away Team Name']
    year = row['Year']
    
    home_stage = get_stage(home_country, year)
    away_stage = get_stage(away_country, year)
    
    overall_performance.append(stage_dict[home_stage])
    overall_performance.append(stage_dict[away_stage])


average_overall_performance = sum(overall_performance) / len(overall_performance)


print(f"Average stage reached by host countries: {average_host_performance}")
print(f"Average stage reached by all countries: {average_overall_performance}")
print(f"Host countries generally perform better than expected: {average_host_performance < average_overall_performance}")


# In[ ]:





# ## Match and Player Performance

# ## 3.Match Outcomes :

# ### a.which match had the most goals scored

# #### Dataset :' WorldCupMatches'

# In[12]:


matches = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupMatches.csv")


# In[13]:


matches['Total Goals'] = matches['Home Team Goals'] + matches['Away Team Goals']

top_matches = matches.sort_values(by='Total Goals', ascending=False).head(10)

top_matches_display = top_matches[['Year', 'Datetime', 'Stage', 'Home Team Name', 'Away Team Name', 'Home Team Goals', 'Away Team Goals', 'Total Goals']]
print(top_matches_display)


# 
# 
# 
# 

# ## 4. Player Achievements :

# #### a.who are the all-time top goal scores in World Cup History

# ##### Dataset : ' WorldCupPlayers '

# In[19]:


players = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupPlayers.csv")


print(players.head())

# Filter rows where the 'Event' column indicates a goal
goal_events = players[players['Event'].str.startswith('G', na=False)]

# Count the number of goals scored by each player
top_scorers = goal_events['Player Name'].value_counts().reset_index()
top_scorers.columns = ['Player Name', 'Goals']

# Sort and display the top goal scorers
top_scorers = top_scorers.sort_values(by='Goals', ascending=False).head(10)
print(top_scorers)


# ## 5.Goal Analysis

# #### a. What is the average number of goals scored per match in different World Cups

# #####  Dataset : ' WorldCupMatches'

# In[8]:


matches = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupMatches.csv")


# In[11]:


matches['Total Goals'] = matches['Home Team Goals'] + matches['Away Team Goals']

average_goals_per_match = matches.groupby('Year')['Total Goals'].mean().reset_index()
average_goals_per_match.columns = ['Year', 'Average Goals per Match']

print(average_goals_per_match)


# #### b.How often do matches go into extra time or penalty shootouts

# ##### Dataset : 'WorldCupMatches'

# In[12]:


matches = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupMatches.csv")


# In[14]:


extra_time_or_penalties = matches[matches['Win conditions'].notna()]

# Count the occurrences of extra time and penalty shootouts
extra_time_count = extra_time_or_penalties[extra_time_or_penalties['Win conditions'].str.contains('extra time', case=False, na=False)].shape[0]
penalty_shootout_count = extra_time_or_penalties[extra_time_or_penalties['Win conditions'].str.contains('penalties', case=False, na=False)].shape[0]

# Total number of matches
total_matches = matches.shape[0]

# Calculate percentages
extra_time_percentage = (extra_time_count / total_matches) * 100
penalty_shootout_percentage = (penalty_shootout_count / total_matches) * 100

# Results
result = {
    "Total Matches": total_matches,
    "Matches with Extra Time": extra_time_count,
    "Matches with Penalty Shootouts": penalty_shootout_count,
    "Percentage of Matches with Extra Time": extra_time_percentage,
    "Percentage of Matches with Penalty Shootouts": penalty_shootout_percentage
}
print(result)


# ## Factors Influencing Wins

# ##  6.Tactical Insights:

# #### a. What formations and tactics are most commonly used by winning teams

# ##### Dataset : ' WorldCupPlayers '

# In[15]:


players = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupPlayers.csv")


# In[21]:


winning_teams_positions = players[players['Position'].notna()]

positions_count = winning_teams_positions.groupby(['Team Initials', 'Position']).size().reset_index(name='Count')

positions_pivot = positions_count.pivot(index='Team Initials', columns='Position', values='Count').fillna(0)

common_formations = positions_pivot.mean().sort_values(ascending=False).reset_index()
common_formations.columns = ['Position', 'Average Count']


plt.figure(figsize=(10, 5))
plt.bar(common_formations['Position'], common_formations['Average Count'], color='green')
plt.xlabel('Position')
plt.ylabel('Average Count of Players')
plt.title('Most Common Player Positions in Winning Teams')
plt.xticks(rotation=45)
plt.show()


# #### b. How does possession percentage correlate with match outcomes

# ##### Dataset: ' WorldCupMatches '

# In[2]:


matches = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupMatches.csv")


# In[9]:


def match_outcome(row):
    if row['Home Team Goals'] > row['Away Team Goals']:
        return 'Home Win'
    elif row['Home Team Goals'] < row['Away Team Goals']:
        return 'Away Win'
    else:
        return 'Draw'

matches['Outcome'] = matches.apply(match_outcome, axis=1)

home_win_goals = matches[matches['Outcome'] == 'Home Win']['Home Team Goals'].mean()
away_win_goals = matches[matches['Outcome'] == 'Away Win']['Away Team Goals'].mean()
draw_goals = matches[matches['Outcome'] == 'Draw'][['Home Team Goals', 'Away Team Goals']].mean().mean()

print(f"Average goals for home wins: {home_win_goals}")
print(f"Average goals for away wins: {away_win_goals}")
print(f"Average goals for draws: {draw_goals}")

labels = ['Home Win', 'Away Win', 'Draw']
values = [home_win_goals, away_win_goals, draw_goals]

plt.figure(figsize=(10, 6))
plt.bar(labels, values, color=['blue', 'orange', 'green'])
plt.xlabel('Match Outcome')
plt.ylabel('Average Goals Scored')
plt.title('Average Goals Scored by Match Outcome')
plt.show()


# ##  7.Key Metrics

# #### a.What are the key metrics (e.g., shots on target, pass completion rate) that influence match outcomes

# ##### Dataset : ' WorldCupMatches '

# In[4]:


matches = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupMatches.csv")


# In[11]:


print(matches.columns)

relevant_metrics = ['Home Team Goals', 'Away Team Goals']

def match_outcome(row):
   if row['Home Team Goals'] > row['Away Team Goals']:
       return 'Home Win'
   elif row['Home Team Goals'] < row['Away Team Goals']:
       return 'Away Win'
   else:
       return 'Draw'

matches['Outcome'] = matches.apply(match_outcome, axis=1)

metrics_avg = {
   'Home Win': matches[matches['Outcome'] == 'Home Win'][relevant_metrics].mean(),
   'Away Win': matches[matches['Outcome'] == 'Away Win'][relevant_metrics].mean(),
   'Draw': matches[matches['Outcome'] == 'Draw'][relevant_metrics].mean()
}

metrics_avg_df = pd.DataFrame(metrics_avg).reset_index()
metrics_avg_df.columns = ['Metric', 'Home Win', 'Away Win', 'Draw']


print(metrics_avg_df)

metrics_avg_df.plot(kind='bar', x='Metric', figsize=(14, 7))
plt.xlabel('Metric')
plt.ylabel('Average Value')
plt.title('Average Values of Key Metrics by Match Outcome')
plt.xticks(rotation=0)
plt.show()


# ## 8. Geographical and Economic Factors:

# #### How does the performance of teams vary by continent

# ##### Dataset:'  WorldCups and WorldCupMatches '

# In[13]:


worldcups = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCups.csv")
matches = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupMatches.csv")


# In[14]:


country_to_continent = {
    'Argentina': 'South America',
    'Brazil': 'South America',
    'Germany': 'Europe',
    'Italy': 'Europe',
    'France': 'Europe',
    'Uruguay': 'South America',
    'England': 'Europe',
    'Spain': 'Europe',
    'Netherlands': 'Europe',
    'Portugal': 'Europe',
    'Sweden': 'Europe',
    'Mexico': 'North America',
    'USA': 'North America',
    'Japan': 'Asia',
    'South Korea': 'Asia',
    'Australia': 'Oceania',
    
}


def get_continent(team):
    return country_to_continent.get(team, 'Other')


matches['Home Team Continent'] = matches['Home Team Name'].apply(get_continent)
matches['Away Team Continent'] = matches['Away Team Name'].apply(get_continent)

continent_metrics = {
    'Continent': [],
    'Wins': [],
    'Draws': [],
    'Losses': [],
    'Goals Scored': [],
    'Goals Conceded': []
}

for continent in set(country_to_continent.values()):
    home_wins = matches[(matches['Home Team Continent'] == continent) & (matches['Home Team Goals'] > matches['Away Team Goals'])].shape[0]
    away_wins = matches[(matches['Away Team Continent'] == continent) & (matches['Away Team Goals'] > matches['Home Team Goals'])].shape[0]
    draws = matches[((matches['Home Team Continent'] == continent) | (matches['Away Team Continent'] == continent)) & (matches['Home Team Goals'] == matches['Away Team Goals'])].shape[0]
    home_goals = matches[matches['Home Team Continent'] == continent]['Home Team Goals'].sum()
    away_goals = matches[matches['Away Team Continent'] == continent]['Away Team Goals'].sum()
    home_conceded = matches[matches['Home Team Continent'] == continent]['Away Team Goals'].sum()
    away_conceded = matches[matches['Away Team Continent'] == continent]['Home Team Goals'].sum()
    
    continent_metrics['Continent'].append(continent)
    continent_metrics['Wins'].append(home_wins + away_wins)
    continent_metrics['Draws'].append(draws)
    continent_metrics['Losses'].append((matches['Home Team Continent'] == continent).sum() + (matches['Away Team Continent'] == continent).sum() - home_wins - away_wins - draws)
    continent_metrics['Goals Scored'].append(home_goals + away_goals)
    continent_metrics['Goals Conceded'].append(home_conceded + away_conceded)

continent_metrics_df = pd.DataFrame(continent_metrics)


print(continent_metrics_df)


continent_metrics_df.set_index('Continent').plot(kind='bar', figsize=(14, 7))
plt.xlabel('Continent')
plt.ylabel('Count')
plt.title('Performance Metrics by Continent')
plt.xticks(rotation=45)
plt.show()


# ##  9. Anecdotal and Memorable Moments:

# ### Historic Moments

# #### a.What are some of the most iconic goals in World Cup history

# In[15]:


players = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupPlayers.csv")


# In[17]:


print(players.columns)

# Filter rows with goal events
goal_events = players[players['Event'].notna() & players['Event'].str.contains('G')].copy()

# Extract the goal details
goal_events['Goal Details'] = goal_events['Event'].apply(lambda x: x.split('G')[1])

# Display the notable goal events
print(goal_events[['Player Name', 'Event', 'Goal Details']].head(10))


# #### b. Which matches are considered the greatest upsets in World Cup history

# ##### Dataset: ' WorldCupMatches '

# In[2]:


matches = pd.read_csv(r"C:\Users\iamke\OneDrive\Desktop\FIFA WC data\WorldCupMatches.csv")


# In[6]:


print(matches.head())


strong_teams = [
    'Brazil', 'Germany', 'Italy', 'Argentina', 
    'France', 'Uruguay', 'England', 'Spain'
]


def is_upset(row):
    home_team = row['Home Team Name']
    away_team = row['Away Team Name']
    home_goals = row['Home Team Goals']
    away_goals = row['Away Team Goals']
    
    if home_team in strong_teams and away_team not in strong_teams and away_goals > home_goals:
        return True
    if away_team in strong_teams and home_team not in strong_teams and home_goals > away_goals:
        return True
    return False


matches['Is Upset'] = matches.apply(is_upset, axis=1)


upsets = matches[matches['Is Upset'] == True]


print(upsets[['Year', 'Datetime', 'Stage', 'Home Team Name', 'Away Team Name', 'Home Team Goals', 'Away Team Goals']].head(10))


# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




