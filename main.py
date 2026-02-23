
#main

import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.indicators  import add_indicators
from src.strategy    import generate_signals, run_backtest, compute_metrics

train, test = load_data('data/btc_project_train.csv', 'data/btc_project_test.csv')
train = add_indicators(train)
test  = add_indicators(test)

train_sig = generate_signals(train)
equity_curve, trades = run_backtest(train_sig)
metrics = compute_metrics(equity_curve, trades)

print("Train baseline metrics:")
for k, v in metrics.items():
    print(f"  {k:<15}: {v}")