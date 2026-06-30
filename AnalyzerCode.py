import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

rankings = pd.read_csv("fifa_mens_rank.csv")
matches = pd.read_csv("matches_1930_2022.csv")

rankings["date"] = pd.to_datetime(rankings["date"])

rankings = rankings[["date", "team", "rank"]]

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

matches = matches[
    matches["Year"].isin(
        TARGET_YEARS
    )
]

print(rankings.shape)
print(matches.shape)

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

team_results = build_finish_table(
    matches
)

team_results.head()

def get_snapshot(
    rankings,
    matches,
    year
):

    world_cup_start = pd.to_datetime(
        matches[
            matches["Year"] == year
        ]["Date"].min()
    )

    snapshot_date = (
        rankings[
            rankings["date"]
            < world_cup_start
        ]["date"]
        .max()
    )

    return (
        rankings[
            rankings["date"]
            <= snapshot_date
        ]
        .sort_values("date")
        .groupby("team")
        .tail(1)
    )

master_rows = []

for year in TARGET_YEARS:

    snapshot = get_snapshot(
        rankings,
        matches,
        year
    )

    merged = (
        team_results[
            team_results["Year"]
            == year
        ]
        .merge(
            snapshot[
                ["team", "rank"]
            ],
            on="team",
            how="left"
        )
    )

    master_rows.append(
        merged
    )

dataset = pd.concat(
    master_rows,
    ignore_index=True
)

dataset.head()

dataset["Prediction_Error"] = abs(
    dataset["rank"]
    - dataset["Finish"]
)


dataset["Team_Year"] = (
    dataset["team"]
    + " ("
    + dataset["Year"].astype(str)
    + ")"
)

dataset.head()

dataset["Prediction_Error"] = abs(
    dataset["rank"]
    - dataset["Finish"]
)


dataset["Team_Year"] = (
    dataset["team"]
    + " ("
    + dataset["Year"].astype(str)
    + ")"
)

dataset.head()

fig = px.scatter(
    dataset,
    x="rank",
    y="Finish",
    color="Year",
    hover_name="team",
    hover_data=[
        "Prediction_Error"
    ],
    title="FIFA Ranking vs World Cup Finish"
)

fig.update_yaxes(
    autorange="reversed"
)

fig.show()

top = (
    dataset
    .sort_values(
        "Prediction_Error",
        ascending=False
    )
    .head(15)
)

plt.figure(figsize=(10,8))

sns.barplot(
    data=top,
    x="Prediction_Error",
    y="Team_Year"
)

plt.title(
    "When FIFA Got It Wrong"
)

plt.show()

corrs = []

for year in TARGET_YEARS:

    subset = dataset[
        dataset["Year"] == year
    ]

    corr = subset[
        "rank"
    ].corr(
        subset["Finish"]
    )

    corrs.append([
        year,
        corr
    ])

corr_df = pd.DataFrame(
    corrs,
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

print(
    f"Overall correlation: "
    f"{dataset['rank'].corr(dataset['Finish']):.3f}"
)

print(
    f"Average prediction error: "
    f"{dataset['Prediction_Error'].mean():.2f}"
)
