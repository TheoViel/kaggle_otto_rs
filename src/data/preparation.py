import pandas as pd
from datetime import datetime

def prepare_data(root):
    df = pd.read_csv(root + "train.csv")
    df_test = pd.read_csv(root + "test.csv")

    if "processed" not in df['path'][0]:
        df['path'] = root + 'processed/' + df['path']
    if "processed" not in df_test['path'][0]:
        df_test['path'] = root + 'processed_test/' + df_test['path']

        df['fold'] = -1
    df.loc[df.index[-2000000:], 'fold']= 0

    df_test['fold'] = -1

    return df, df_test
