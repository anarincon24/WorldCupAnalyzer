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

We will load these into our code using pandas, and shorten our 'rankings' data to only the columns we need.
We also need to convert the 'date' column into something that our Python script can interpret. Python currently thinks "2006-05-17" is just a string. After conversion it becomes a true date object that Python can compare.

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

## Step 3: Measure World Cup Performance

To compare rankings with outcomes, we need a way to measure how well each team performed.

We'll assign a score based on the deepest round reached. Translating a team's finish from a string to a number will allow us to work with the data more efficiently. To do this we will define this function:
```
def build_finish_table(matches):

    team_results = []

    for year in TARGET_YEARS:

        wc = matches[
            matches["Year"] == year
        ]

        teams = pd.unique(
            pd.concat([
                wc["home_team"],
                wc["away_team"]
            ])
        )

        for team in teams:

            games = wc[
                (wc["home_team"] == team)
                |
                (wc["away_team"] == team)
            ]

            rounds = games["Round"].unique()

            finish = 32

            if "Final" in rounds:
                finish = 2

            elif "Third-place match" in rounds:
                finish = 4

            elif "Semi-finals" in rounds:
                finish = 4

            elif "Quarter-finals" in rounds:
                finish = 8

            elif "Round of 16" in rounds:
                finish = 16

            team_results.append(
                [year, team, finish]
            )

    return pd.DataFrame(
        team_results,
        columns=[
            "Year",
            "team",
            "Finish"
        ]
    )
```

This function will return a data frame including only the 3 pieces of information that we need for this project, "Year" "team" and "Finish"

Now, lets run the function and take a look at what our data looks like utilizing .head()
```
team_results = build_finish_table(matches)

team_results.head()
```
## Step 4: Find Each Team's FIFA Ranking

Now we need to determine each team's FIFA ranking before the tournament began.

The function below finds the most recent FIFA ranking available before each World Cup started.
```
def get_snapshot(
    rankings,
    matches,
    year
):

    world_cup_start = pd.to_datetime(
        matches[ matches["Year"] == year]["Date"].min()
    )

    snapshot_date = (
        rankings[rankings["date"]< world_cup_start]["date"].max()
    )

    return (
        rankings[rankings["date"]<= snapshot_date].sort_values("date").groupby("team").tail(1)
    )
```
## Step 5: Build the Dataset

Now we can combine everything into one dataset.

For each World Cup:

Find the pre-tournament rankings
Find each team's finish score
Merge the information together
```
master_rows = []

for year in TARGET_YEARS:

    snapshot = get_snapshot(
        rankings,
        matches,
        year
    )

    merged = (
        team_results[team_results["Year"]== year].merge(snapshot[["team", "rank"]],
            on="team",
            how="left"
        )
    )

    master_rows.append(merged)
```
```
dataset = pd.concat(
    master_rows,
    ignore_index=True
)

dataset.head()
```
Each row now contains:

- Team
- Year
- FIFA Ranking
- World Cup Finish

This is the dataset we'll use for the rest of the project.

## Step 6: Create Prediction Metrics

Let's create a metric called Prediction Error.
```
dataset["Prediction_Error"] = abs(
    dataset["rank"] -
    dataset["Finish"]
)
```
This measures how far reality differed from expectations.

A small value means FIFA rankings were accurate.

A large value means FIFA rankings were very wrong.

We'll also create labels for plotting:
```
dataset["Team_Year"] = (
    dataset["team"]
    + " ("
    + dataset["Year"].astype(str)
    + ")"
)
```
## Step 7: Quick Summary Statistics

Before building visualizations, let's calculate a few summary metrics.
```
print(f"Teams analyzed: {len(dataset)}")

print(f"World Cups analyzed: "f"{dataset['Year'].nunique()}")

print(f"Average prediction error: "f"{dataset['Prediction_Error'].mean():.2f}")

print(f"Overall correlation: "f"{dataset['rank'].corr(dataset['Finish']):.3f}")
```
The correlation tells us how strongly rankings and outcomes are related.

A value closer to 1 indicates rankings are highly predictive.

A value closer to 0 suggests rankings have little predictive power.

## Step 8: Visualize Ranking vs Performance

Let's compare rankings and outcomes directly with an interactive scatter plot!
```
fig = px.scatter(
    dataset,
    x="rank",
    y="Finish",
    color="Year",
    hover_name="team",
    hover_data=["Prediction_Error"]
    title="FIFA Ranking vs World Cup Finish"
)

fig.update_yaxes(autorange="reversed")

fig.show()
```
Try hovering over points after you run the code to investigate specific teams!
<img width="787" height="560" alt="Screenshot 2026-07-01 001952" src="https://github.com/user-attachments/assets/da0d41f4-c594-42a6-a17d-02cafd9d960d" />

## Step 9: When FIFA Got It Wrong

Even good prediction systems make mistakes.
Let's find the biggest ones.
```
top = (
    dataset
    .sort_values(
        "Prediction_Error",
        ascending=False
    )
    .head(15)
)
```
Then visualize them:
```
sns.barplot(
    data=top,
    x="Prediction_Error",
    y="Team_Year"
)
```
<img width="757" height="522" alt="Screenshot 2026-07-01 001036" src="https://github.com/user-attachments/assets/9cd7cf45-fc4d-4fe1-9ada-11c4fea8def2" />

These teams represent some of the biggest surprises in World Cup history when you compare final tournament placements to Fifa rankings before their respective World Cup.

## Step 10: Have Rankings Improved?

Have FIFA rankings become more accurate over time?
Let's calculate a correlation for each World Cup and visualize it in a graph.
```
corrs = []

for year in TARGET_YEARS:

    subset = dataset[dataset["Year"] == year]

    corr = subset["rank"].corr(subset["Finish"])

    corrs.append([year,corr])

corr_df = pd.DataFrame(corrs,
    columns=[
        "Year",
        "Correlation"
    ]
)

plt.figure(figsize=(10,6))

sns.lineplot(
    data=corr_df,
    x="Year",
    y="Correlation",
    marker="o"
)
plt.ylim(0,1)

plt.title(
    "How Predictive FIFA Rankings Were"
)

plt.show()
```
<img width="760" height="488" alt="Screenshot 2026-07-01 001059" src="https://github.com/user-attachments/assets/b48b6ea1-8643-4ebb-8740-17be81d37c4b" />

If the line trends upward, FIFA rankings may be getting better at predicting success.

## Final Verdict

Let's answer our original question.
```
print(
    f"Overall correlation: "
    f"{dataset['rank'].corr(dataset['Finish']):.3f}"
)

print(
    f"Average prediction error: "
    f"{dataset['Prediction_Error'].mean():.2f}"
)
```
