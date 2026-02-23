
#libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# data
train = pd.read_csv('data/btc_project_train.csv')
test  = pd.read_csv('data/btc_project_test.csv')

train['Datetime'] = pd.to_datetime(train['Datetime'])
test['Datetime']  = pd.to_datetime(test['Datetime'])

train = train.sort_values('Datetime').reset_index(drop=True)
test  = test.sort_values('Datetime').reset_index(drop=True)

# info
for name, df in [('TRAIN', train), ('TEST', test)]:
    print(f"\n{name}")
    print(f"  Rows      : {len(df)}")
    print(f"  Date range: {df['Datetime'].min()}  →  {df['Datetime'].max()}")
    print(f"  Close min : {df['Close'].min():.2f}")
    print(f"  Close max : {df['Close'].max():.2f}")
    print(f"  Missing   : {df.isnull().sum().to_dict()}")


#chart
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(train['Datetime'], train['Close'], linewidth=0.6, color='steelblue')
axes[0].set_title('Train Set — BTC/USDT 5min')
axes[0].set_ylabel('Price (USDT)')

axes[1].plot(test['Datetime'], test['Close'], linewidth=0.6, color='darkorange')
axes[1].set_title('Test Set — BTC/USDT 5min')
axes[1].set_ylabel('Price (USDT)')

plt.tight_layout()
plt.savefig('results/eda_price.png', dpi=150)
plt.show()
