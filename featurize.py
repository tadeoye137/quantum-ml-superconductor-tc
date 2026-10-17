"""
Create featurized_data.csv from the SuperCon dataset using Magpie features.
(Feature-engineering part of my undergraduate XGBoost project.)

Input:  a CSV with columns "material" (chemical formula) and "critical_temp" (K)
Output: featurized_data.csv with 132 Magpie features + critical_temp
"""
import pandas as pd
from matminer.featurizers.conversions import StrToComposition
from matminer.featurizers.composition import ElementProperty

# 1. Load data: keep only the formula and Tc
df = pd.read_csv("superconJ.csv")[["critical_temp", "material"]]
print(f"Dataset size: {len(df)} materials")

# 2. Convert formula strings (e.g. "YBa2Cu3O7") into composition objects
df = StrToComposition().featurize_dataframe(df, "material")

# 3. Generate the 132 Magpie elemental-property statistics for each composition
ep = ElementProperty.from_preset("magpie")
df = ep.featurize_dataframe(df, "composition")

# 4. Save
df.to_csv("featurized_data.csv", index=False)
print("Saved featurized_data.csv")
