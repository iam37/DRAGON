from pathlib import Path
import pandas as pd

def load_data_dir(data_dir, slug=None, split=None):
    """Loads and returns pandas dataframe"""

    data_dir = Path(data_dir)

    if split:
        catalog = data_dir / f"splits/{slug}-{split}.csv"
    else:
        catalog = data_dir / "info.csv"

    return pd.read_csv(catalog)
