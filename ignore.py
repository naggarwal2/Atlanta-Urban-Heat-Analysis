import pandas as pd
file_path = "/Users/sydneelear/Downloads/cleaned_data.csv"
data = pd.read_csv(file_path)
data.describe(include = 'all')