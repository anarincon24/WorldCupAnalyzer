# Analyzing FIFA Rankings & World Cup Success with Data Visualization

## Introduction
Every four years, millions of soccer fans try to predict who will lift the FIFA World Cup trophy.

One of the most popular ways to compare teams is the FIFA World Ranking system. Teams are ranked based on their international performances, with stronger teams receiving higher rankings.

But how useful are these rankings? Can FIFA rankings actually predict how teams perform at the World Cup?

In this project, we'll analyze every FIFA World Cup from 1994 to 2022 and investigate whether highly ranked teams consistently finish higher in the tournament. We will do this using these Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- plotly

## Step 1: Import the Data
We will start by importing these libraries that we will using throughout the project:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
```
Next, we need to download and load the datasets that we will be using in our code. We will be using two datasets from Kaggle providing us with
[FIFA Men's Teams Rankings](https://www.kaggle.com/datasets/lucasyukioimafuko/fifa-mens-world-ranking?utm_source=chatgpt.com) and all [Matches from 1930 to 2022](https://www.kaggle.com/datasets/piterfm/fifa-football-world-cup?utm_source=chatgpt.com&select=matches_1930_2022.csv)

The datasets in CSV form can be downloaded from the project github instead of going through Kaggle.

We will load these into our code using pandas, convert the date column into something that python can interpret, and shorten our 'rankings' data to only the columns we need.

```
rankings = pd.read_csv("fifa_mens_rank.csv")
matches = pd.read_csv("matches_1930_2022.csv")

rankings["date"] = pd.to_datetime(rankings["date"])

rankings = rankings[["date", "team", "rank"]]
```
## Step 2: Focus on the FIFA Ranking Era

The FIFA World Ranking system was introduced in 1992.

Since we're trying to evaluate FIFA rankings, it only makes sense to analyze tournaments that happened after rankings existed. 
```
TARGET_YEARS = [
    1994,
    1998,
    2002,
    2006,
    2010,
    2014,
    2018,
    2022
]
```
Filter the matches dataset to only these target years, and our .shape commands will show us how much data is left for us to work with in terms of (rows, columns).
```
matches = matches[
    matches["Year"].isin(
        TARGET_YEARS
    )
]
print(rankings.shape)
print(matches.shape)
```

