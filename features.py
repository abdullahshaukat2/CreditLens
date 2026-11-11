import numpy as np
import pandas as pd

PAY_STATUS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYMENTS = [f"PAY_AMT{i}" for i in range(1, 7)]

def add_features(df):
    out = df.copy()
    limit = out["LIMIT_BAL"].clip(lower=1)
    util = out[BILLS].div(limit, axis=0)
    out["util_avg_6m"] = util.mean(axis=1)
    out["util_max_6m"] = util.max(axis=1)
    
    status = out[PAY_STATUS]
    out["months_late_6m"] = (status >= 1).sum(axis=1)
    out["max_delay_6m"] = status.max(axis=1).clip(lower=0)

    prev_bill = out["BILL_AMT2"]
    ratio = np.where(prev_bill > 0, out["PAY_AMT1"] / prev_bill.clip(lower=1), 1.0)
    out["pay_ratio_latest"] = np.clip(ratio, 0, 2)
    return out    