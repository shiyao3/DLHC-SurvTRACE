from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
from utils import LabelTransform


def load_support_data(path, quantiles):

    get_target = lambda df: (df['duration'].values, df['event'].values)

    df = pd.read_hdf(path)

    times = np.quantile(df["duration"][df["event"] == 1.0], quantiles).tolist()
    cols_categorical = ["x1", "x2", "x3", "x4", "x5", "x6"]
    cols_standardize = ['x0', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

    df_feat = df.drop(["duration", "event"], axis=1)
    df_feat_standardize = df_feat[cols_standardize]
    df_feat_standardize_disc = StandardScaler().fit_transform(df_feat_standardize)
    df_feat_standardize_disc = pd.DataFrame(df_feat_standardize_disc, columns=cols_standardize)

    df_feat = pd.concat([df_feat[cols_categorical], df_feat_standardize_disc], axis=1)

    vocab_size = 0
    for i, feat in enumerate(cols_categorical):
        df_feat[feat] = LabelEncoder().fit_transform(df_feat[feat]).astype(float) + vocab_size
        vocab_size = df_feat[feat].max() + 1

    # get the largest duraiton time
    max_duration_idx = df["duration"].argmax()
    df_test = df_feat.drop(max_duration_idx).sample(frac=0.3)
    df_train = df_feat.drop(df_test.index)
    df_val = df_train.drop(max_duration_idx).sample(frac=0.1)
    df_train = df_train.drop(df_val.index)

    # assign cuts
    # labtrans = LabTransDiscreteTime(cuts=np.array([0]+times+[df["duration"].max()]))
    labtrans = LabelTransform(cuts=np.array([0] + times + [df["duration"].max()]))

    labtrans.fit(*get_target(df.loc[df_train.index]))
    # y = labtrans.fit_transform(*get_target(df)) # y = (discrete duration, event indicator)
    y = labtrans.transform(*get_target(df))  # y = (discrete duration, event indicator)
    df_y_train = pd.DataFrame(
        {"duration": y[0][df_train.index], "event": y[1][df_train.index], "proportion": y[2][df_train.index]},
        index=df_train.index)
    df_y_val = pd.DataFrame(
        {"duration": y[0][df_val.index], "event": y[1][df_val.index], "proportion": y[2][df_val.index]},
        index=df_val.index)
    # df_y_test = pd.DataFrame({"duration": y[0][df_test.index], "event": y[1][df_test.index], "proportion":y[2][df_test.index]}, index=df_test.index)
    df_y_test = pd.DataFrame({"duration": df['duration'].loc[df_test.index], "event": df['event'].loc[df_test.index]})