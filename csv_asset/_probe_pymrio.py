import pymrio
import pandas as pd
import inspect

# Parse EORA with correct args
print("=== Parsing EORA 2015 ===")
eora = pymrio.parse_eora26('csv_asset/mrio/unzipped/Eora26_2015_bp', year=2015)
print("Parsed OK. Type:", type(eora))

# Aggregation-related methods on the object
print("\n=== eora aggregate-related methods ===")
print([x for x in dir(eora) if 'agg' in x.lower() or 'concord' in x.lower()])

# Index structure
print("\n=== eora.Z.index levels (first 5) ===")
print(eora.Z.index[:5])
print("\n=== eora.Z unique countries (first 10) ===")
countries = eora.Z.index.get_level_values(0).unique()
print(countries[:10].tolist())
print("Total countries:", len(countries))

# Check build_agg_vec / build_agg_matrix signatures
print("\n=== build_agg_vec signature ===")
print(inspect.signature(pymrio.build_agg_vec))

print("\n=== build_agg_matrix signature ===")
print(inspect.signature(pymrio.build_agg_matrix))

# Check aggregate method signature if it exists
if hasattr(eora, 'aggregate'):
    print("\n=== eora.aggregate signature ===")
    print(inspect.signature(eora.aggregate))
