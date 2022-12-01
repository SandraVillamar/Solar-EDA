from math import floor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
This file provides function to generate 
kernel density plots of power outputs according 
to the climate assigned to each location. The kde plot
is seperated by its climate type and season (spring,
summer, autumn, winter). In each plot, hue is set
to be the locations within the given climate type.

The purpose of do so is to show taht locations with the
same climate type tend to have similar patterns in terms of 
their kernel density plots. Which explains why fitting model
with climate information helps improve model's performance.

Zhenduo Wen
"""


def climate_kde_plot(df: pd.DataFrame, climate_dict: dict) -> None:
    """
    Function Usage:
    divide df by each rows location's climate type.
    for each season (spring, summer, autumn, winter), generate
    corresponding climate's kernel density plot

    :param df: pd.DataFrame, cleaned dataframe to generate
    kernel density plot
    :param climate_dict: dict, contains the classification from
    location to climate type.
    Usage: key: str, climate_name, value: list, contains strings of
    location names
    :return: None, generate a plot.
    """
    assert isinstance(climate_dict, dict)

    # for different seasons
    f, axes = plt.subplots(len(climate_dict.keys()), 4, figsize=(25, 30))
    ax_idx = 0
    seasons = df['season'].unique()

    # plot kernel densities
    for climate_name in climate_dict.keys():
        climate_locations = climate_dict[climate_name]
        for season in seasons:
            x = floor(ax_idx / 4)
            y = ax_idx % 4

            # may want to add parameter: multiple='stack'
            sub_plot = sns.kdeplot(data=df[(df['location'].isin(climate_locations)) & (df['season'] == season)], \
                                   x='polypwr', hue='location',
                                   ax=axes[x, y], bw_adjust=1)
            sub_plot.set_title(climate_name + " in " + season)
            # sub_plot.set(ylim=(0, 0.15))
            sub_plot.set(xlim=(0, 35))
            ax_idx += 1

    plt.show()


# local plot
if __name__ == "__main__":
    # load dataframe
    df = pd.read_csv("Pasion et al dataset.csv")

    # convert column names to lowercase
    df.columns = df.columns.str.lower()
    dummy = pd.get_dummies(df.season, prefix='season')
    df_with_season_code = pd.concat([df, dummy], axis=1)
    df.columns = df.columns.str.replace('.', '_')
    df = df.rename(columns={'ambienttemp': 'ambient_temp'})

    # normalize columns
    numeric_cols = ['humidity', 'ambient_temp', 'wind_speed',
                    'pressure', 'cloud_ceiling', 'latitude', 'longitude', 'altitude', 'month']
    df[numeric_cols] = (df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std()

    # one hot encoding
    dummy = pd.get_dummies(df.season, prefix='season')
    df_with_season_code = pd.concat([df, dummy], axis=1)

    # assign climate type to each location
    climate_dict = {'hot-dry': ['March AFB', 'Travis'],
                    'cold-dry': ['Hill Weber', 'USAFA', 'Peterson', 'Offutt'],
                    'cold-humid': ['Grissom', 'Malmstrom', 'MNANG', 'Camp Murray'],
                    'hot-humid': ['JDMT', 'Kahului'],
                    'mixture': df.location.unique()
                    }

    # generate kernel density plot
    climate_kde_plot(df_with_season_code, climate_dict)
