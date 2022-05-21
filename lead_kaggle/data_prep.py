from cmath import pi
import pandas as pd


def pipeline(df):
    """
    should take a dataframe, clean + add features and then write to parquet.  Should apply to both train and test.

    """
    # df = df.dropna(subset="meter_reading")

    df["24h_avg"] = (
        df.sort_values(by=["timestamp"], ascending=True)
        .groupby("building_id")["meter_reading"]
        .rolling(24, 3)
        .mean()
        .reset_index(drop=True, level=0)
    )
    df["24h_std"] = (
        df.sort_values(by=["timestamp"], ascending=True)
        .groupby("building_id")["meter_reading"]
        .rolling(24, 3)
        .std()
        .reset_index(drop=True, level=0)
    )
    df["1w_avg"] = (
        df.sort_values(by=["timestamp"], ascending=True)
        .groupby("building_id")["meter_reading"]
        .rolling(168, 24)
        .mean()
        .reset_index(drop=True, level=0)
    )
    df["1w_std"] = (
        df.sort_values(by=["timestamp"], ascending=True)
        .groupby("building_id")["meter_reading"]
        .rolling(168, 24)
        .std()
        .reset_index(drop=True, level=0)
    )
    df["minus_24h_val"] = (
        df.sort_values(by=["timestamp"], ascending=True)
        .groupby("building_id")["meter_reading"]
        .shift(24, fill_value=0)
    )
    df["minus_week_val"] = (
        df.sort_values(by=["timestamp"], ascending=True)
        .groupby("building_id")["meter_reading"]
        .shift(168, fill_value=0)
    )

    def z_score(row):
        if row["24h_std"] > 0.0:
            return (row["meter_reading"] - row["24h_avg"]) / row["24h_std"]
        else:
            return 0.0

    df["z_score"] = df.apply(z_score, axis=1)
    df = df.fillna(0)

    return df


if __name__ == "__main__":

    df_train = pd.read_csv(
        "/Users/TimothyW/Fun/energy-kaggle/data/train_features.csv",
        parse_dates=["timestamp"],
    )

    pipeline(df_train).to_parquet("data/prepped_train_features.parquet")

    df_test = pd.read_csv(
        "/Users/TimothyW/Fun/energy-kaggle/data/test_features.csv",
        parse_dates=["timestamp"],
    )

    pipeline(df_test).to_parquet("data/prepped_test_features.parquet")
