from ast import Import
from venv import create
import matplotlib
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import RocCurveDisplay
from catboost import (
    CatBoostClassifier,
    Pool,
    EShapCalcType,
    EFeaturesSelectionAlgorithm,
)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import shap
import numpy as np
import ipywidgets

PLOTS = {}

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    from imblearn.over_sampling import SMOTE


def create_submission(clf, test_set_features, f):
    X = test_set_features[f]
    print(X.info())
    preds = clf.predict(X)
    df_submit = test_set_features[["row_id"]]
    df_submit["anomaly"] = preds
    return df_submit


def compare_train_test(clf, X_train, y_train, X_test, y_test, bins=30):
    fig = plt.figure()
    decisions = []
    for X, y in ((X_train, y_train), (X_test, y_test)):
        d1 = clf.predict_proba(
            X[
                (y > 0.5).reshape(
                    y.shape[0],
                )
            ]
        )[:, 1]
        d2 = clf.predict_proba(
            X[
                (y < 0.5).reshape(
                    y.shape[0],
                )
            ]
        )[:, 1]
        decisions += [d1, d2]

    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    plt.hist(
        decisions[0],
        color="r",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="Anom (train)",
    )
    plt.hist(
        decisions[1],
        color="b",
        alpha=0.5,
        range=low_high,
        bins=bins,
        histtype="stepfilled",
        density=True,
        label="Norm (train)",
    )

    hist, bins = np.histogram(decisions[2], bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    width = bins[1] - bins[0]
    center = (bins[:-1] + bins[1:]) / 2
    plt.errorbar(center, hist, yerr=err, fmt="o", c="r", label="Anom (test)")

    hist, bins = np.histogram(decisions[3], bins=bins, range=low_high, normed=True)
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    plt.errorbar(center, hist, yerr=err, fmt="o", c="b", label="Normal (test)")

    plt.xlabel("BDT output")
    plt.ylabel("Arbitrary units")
    plt.legend(loc="best")

    return fig


if __name__ == "__main__":

    df = pd.read_parquet(
        "/Users/TimothyW/Fun/energy-kaggle/data/prepped_train_features.parquet"
    )

    num_features = [
        "meter_reading",
        "air_temperature",
        "cloud_coverage",
        "dew_temperature",
        "precip_depth_1_hr",
        "sea_level_pressure",
        "wind_direction",
        "wind_speed",
        "24h_avg",
        "24h_std",
        "z_score",
        "square_feet",
        # "year_built",
        "floor_count",
        "air_temperature_mean_lag7",
        "air_temperature_std_lag7",
        "minus_24h_val",
        "minus_week_val",
        "1w_avg",
        "1w_std",
        # "hour",
        "weekday",
    ]
    cat_features = []
    features = num_features + cat_features

    X = df[features]
    y = df[["anomaly"]]

    group_kf = GroupKFold(n_splits=3)
    groups = df.building_id.values

    clfs = []
    summaries = []
    idx = 1
    for train_index, test_index in group_kf.split(X, y, groups=groups):
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]

        SandM = SMOTE(random_state=69)

        X_train_smote, y_train_smote = SandM.fit_resample(X_train, y_train)
        print(" SMOTE done, now training ")

        train_data = Pool(
            data=X_train_smote,
            feature_names=features,
            label=y_train_smote,
            cat_features=cat_features,
        )

        test_data = Pool(
            data=X_test, feature_names=features, label=y_test, cat_features=cat_features
        )

        clf = CatBoostClassifier(
            iterations=1000, early_stopping_rounds=15, eval_metric="AUC"
        )

        clf.fit(train_data, eval_set=test_data, verbose=True)

        clfs.append(clf)

        PLOTS[f"sep_plot_{idx}"] = compare_train_test(
            clf, X_train, y_train, X_test, y_test
        )
        predictions = clf.predict(X_test)

        print("ROC SCORE = ", roc_auc_score(y_test, predictions))

        fig = plt.figure()
        ax = fig.gca()
        ax.cla()
        RocCurveDisplay.from_predictions(y_test, predictions, ax=ax)

        PLOTS[f"roc_curve_{idx}"] = fig

        idx = idx + 1

    for k, v in PLOTS.items():
        v.savefig(f"{k}.png")

    # df_test = pd.read_parquet("data/prepped_test_features.parquet")

    # df_submit = create_submission(
    #   clfs[0], df_test, summaries[0]["selected_features_names"]
    # )

# df_submit.to_csv("submission_4.csv", index=0)

# FEATURE importances
# classifier plot
