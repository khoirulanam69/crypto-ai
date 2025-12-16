import numpy as np

def build_state(df):
    returns = df["close"].pct_change().fillna(0)
    vol = returns.rolling(10).std().fillna(0)

    state = np.concatenate([
        returns.values[-20:],
        vol.values[-20:]
    ])

    return state
