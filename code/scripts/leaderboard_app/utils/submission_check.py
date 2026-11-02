import numpy as np
import pandas as pd

def check_before_submission(
    df,
):
    if not "prediction" in df.columns:
      raise ValueError(f"The submitted df does not contain a 'prediction' column. Make sure to name it accordingly.")
    if not len(df)==5084:
      raise ValueError(f"The submitted df length ({len(df)}) does not match the length of the test set ({5084}).")
    if not np.all([p in [0,1] for p in df["prediction"]]):
      raise ValueError(f"The submitted predictions contain unknown labels please make sure to respect the format. Prediction should be integers: 0 or 1.")

    return True