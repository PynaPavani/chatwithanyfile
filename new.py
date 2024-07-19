import pandas as pd

COLUMN_NAMES = ['query', 'response', 'chartname', 'file_type']
df = pd.DataFrame(columns=COLUMN_NAMES)
df.to_csv('metadata.csv', index=False)