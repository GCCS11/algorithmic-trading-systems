
#main

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.data_loader import load_data
from src.indicators  import add_indicators
from src.strategy    import generate_signals, run_backtest, compute_metrics
from src.optimizer   import walk_forward

# ── Load and prepare ───────────────────────────────────────────────────────────
train, test = load_data('data/btc_project_train.csv', 'data/btc_project_test.csv')
train = add_indicators(train)
test  = add_indicators(test)

# ── Walk-forward optimization (runs once, then loads from file) ────────────────
if os.path.exists('results/walk_forward_results.csv') and os.path.exists('results/best_params_log.json'):
    print("Loading saved walk-forward results...")
    results_df = pd.read_csv('results/walk_forward_results.csv')
    with open('results/best_params_log.json', 'r') as f:
        params_log = json.load(f)
else:
    print("Running walk-forward optimization...")
    results, params_log = walk_forward(train, train_months=1, test_weeks=1, n_trials=100)
    results_df = pd.read_csv('results/walk_forward_results.csv')

# ── Walk-forward summary ───────────────────────────────────────────────────────
print(f"\nWalk-forward summary:")
print(f"  Windows total    : {len(results_df)}")
print(f"  Windows positive : {(results_df['calmar'] > 0).sum()}")
print(f"  Avg Calmar       : {results_df['calmar'].mean():.4f}")
print(f"  Avg Sharpe       : {results_df['sharpe'].mean():.4f}")
print(f"  Avg Win Rate     : {results_df['win_rate'].mean():.2%}")
print(f"  Avg Trades/window: {results_df['total_trades'].mean():.1f}")

# ── Best params from highest Calmar window ────────────────────────────────────
best_idx    = results_df['calmar'].idxmax()
best_params = params_log[best_idx]
print(f"\nBest params (window: {results_df.loc[best_idx, 'window_start']} -> {results_df.loc[best_idx, 'window_end']}):")
for k, v in best_params.items():
    print(f"  {k}: {v}")

# ── Final evaluation on test set ──────────────────────────────────────────────
print("\nTest set evaluation...")
test_sig         = generate_signals(test, rsi_ob=best_params['rsi_ob'], rsi_os=best_params['rsi_os'])
test_eq, test_tr = run_backtest(test_sig,
                                atr_mult=best_params['atr_mult'],
                                take_profit_mult=best_params['take_profit_mult'],
                                risk_pct=best_params['risk_pct'],
                                cooldown=best_params['cooldown'])
test_metrics = compute_metrics(test_eq, test_tr)

print("Test set metrics:")
for k, v in test_metrics.items():
    print(f"  {k:<15}: {v}")



# ── Portfolio value chart ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))

# Plot each segment separately to show the gap
ax.plot(test_sig['Datetime'].values[:270], test_eq[:270],
        linewidth=0.8, color='steelblue', label='Equity')
ax.plot(test_sig['Datetime'].values[271:], test_eq[271:],
        linewidth=0.8, color='steelblue')

# Mark the gap
ax.axvline(test_sig['Datetime'].iloc[270], color='orange',
           linewidth=1.5, linestyle='--', label='Data gap (122 days)')

ax.set_title('Portfolio Value — Test Set')
ax.set_ylabel('Equity (USDT)')
ax.set_xlabel('Date')
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig('results/portfolio_test.png', dpi=150)
plt.show()


# ── Returns table ─────────────────────────────────────────────────────────────
test_sig = test_sig.copy()
test_sig['equity'] = test_eq
test_sig['ret']    = test_sig['equity'].pct_change().fillna(0)
test_sig['month']  = test_sig['Datetime'].dt.to_period('M')
test_sig['quarter']= test_sig['Datetime'].dt.to_period('Q')
test_sig['year']   = test_sig['Datetime'].dt.to_period('Y')

monthly   = test_sig.groupby('month')['ret'].apply(lambda x: (1+x).prod()-1).reset_index()
quarterly = test_sig.groupby('quarter')['ret'].apply(lambda x: (1+x).prod()-1).reset_index()
annually  = test_sig.groupby('year')['ret'].apply(lambda x: (1+x).prod()-1).reset_index()

monthly.columns   = ['Period', 'Return']
quarterly.columns = ['Period', 'Return']
annually.columns  = ['Period', 'Return']

print("\nMonthly returns:")
for _, row in monthly.iterrows():
    print(f"  {row['Period']}  {row['Return']:>8.2%}")

print("\nQuarterly returns:")
for _, row in quarterly.iterrows():
    print(f"  {row['Period']}  {row['Return']:>8.2%}")

print("\nAnnual returns:")
for _, row in annually.iterrows():
    print(f"  {row['Period']}  {row['Return']:>8.2%}")

monthly.to_csv('results/monthly_returns.csv', index=False)
quarterly.to_csv('results/quarterly_returns.csv', index=False)
annually.to_csv('results/annual_returns.csv', index=False)


# ── Sensitivity analysis ±20% ─────────────────────────────────────────────────
print("\nSensitivity analysis (±20% on best params):")
print(f"  {'Parameter':<20} {'Base':>10} {'-20%':>10} {'+20%':>10}")
print(f"  {'-'*52}")

base_calmar = test_metrics['calmar']

for param, value in best_params.items():
    results_row = {}
    for pct, label in [(-0.2, '-20%'), (0.2, '+20%')]:
        modified = dict(best_params)
        if param in ['rsi_ob', 'rsi_os', 'cooldown']:
            modified[param] = int(round(value * (1 + pct)))
        else:
            modified[param] = value * (1 + pct)

        sig_    = generate_signals(test, rsi_ob=modified['rsi_ob'], rsi_os=modified['rsi_os'])
        eq_, tr_ = run_backtest(sig_,
                                atr_mult=modified['atr_mult'],
                                take_profit_mult=modified['take_profit_mult'],
                                risk_pct=modified['risk_pct'],
                                cooldown=modified['cooldown'])
        m_ = compute_metrics(eq_, tr_)
        results_row[label] = m_['calmar']

    print(f"  {param:<20} {base_calmar:>10.4f} {results_row['-20%']:>10.4f} {results_row['+20%']:>10.4f}")


# ── Walk-forward Calmar chart ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 4))
ax.bar(range(len(results_df)), results_df['calmar'].clip(-40, 40),
       color=['steelblue' if c > 0 else 'salmon' for c in results_df['calmar']])
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('Walk-Forward Calmar Ratio by Window')
ax.set_xlabel('Window')
ax.set_ylabel('Calmar Ratio (clipped ±40)')
plt.tight_layout()
plt.savefig('results/walk_forward_calmar.png', dpi=150)
plt.show()


