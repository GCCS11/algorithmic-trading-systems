
from src.data_loader import load_data
from src.indicators  import add_indicators


train, test = load_data('data/btc_project_train.csv', 'data/btc_project_test.csv')
train = add_indicators(train)
test  = add_indicators(test)

print(f"Train: {len(train)} rows  |  {train['Datetime'].min()} -> {train['Datetime'].max()}")
print(f"Test : {len(test)} rows  |  {test['Datetime'].min()} -> {test['Datetime'].max()}")
print(f"\nColumns: {list(train.columns)}")
print(train.head(2).to_string())

