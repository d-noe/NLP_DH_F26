import base64
import io
import pandas as pd

def encode_b64(file_path):
    with open(file_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return encoded

def decode_b64_into_df(b64_encoded_str):
    return pd.read_csv(io.BytesIO(base64.b64decode(b64_encoded_str)))