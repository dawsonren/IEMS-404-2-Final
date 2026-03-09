import pandas as pd
from pymatgen.core import Composition
from matminer.featurizers.composition import ElementProperty

INPUT = "lithium_battery_materials.csv"
OUTPUT = "lithium_battery_materials_augmented.csv"

# these are all the Magpie features. You can choose fewer if you want.
features = ['Electronegativity', 'NValence', 'SpaceGroupNumber']

# these are the statistical quantities we can compute
stats = ['mean', 'range', 'avg_dev']   # we can use any of 'mean', 'minimum', 'maximum', 'range', 'avg_dev', 'mode'

df = pd.read_csv(INPUT)

print(f"Loaded {len(df)} rows. Computing Magpie features...")

df["composition"] = df["formula_pretty"].map(Composition)

ep = ElementProperty(data_source='magpie',features=features, stats=stats)
ep.set_n_jobs(1)  # avoid multiprocessing issues
df = ep.featurize_dataframe(df, col_id="composition", ignore_errors=True)

df = df.drop(columns=["composition"])

df.to_csv(OUTPUT, index=False)
print(f"Saved {len(df)} rows x {len(df.columns)} columns to {OUTPUT}")
