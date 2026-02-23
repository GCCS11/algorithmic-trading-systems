

import numpy as np
import pandas as pd
import optuna
import time
from src.strategy import generate_signals, run_backtest, compute_metrics

optuna.logging.set_verbosity(optuna.logging.WARNING)


class Objective:
    def __init__(self, df):
        self.df = df

    def __call__(self, trial):
        rsi_ob           = trial.suggest_int('rsi_ob', 55, 80)
        rsi_os           = trial.suggest_int('rsi_os', 20, 45)
        atr_mult         = trial.suggest_float('atr_mult', 1.5, 5.0)
        take_profit_mult = trial.suggest_float('take_profit_mult', 1.0, 4.0)
        risk_pct         = trial.suggest_float('risk_pct', 0.002, 0.02)
        cooldown         = trial.suggest_int('cooldown', 5, 50)

        try:
            sig    = generate_signals(self.df, rsi_ob=rsi_ob, rsi_os=rsi_os)
            eq, tr = run_backtest(sig, atr_mult=atr_mult,
                                  take_profit_mult=take_profit_mult,
                                  risk_pct=risk_pct, cooldown=cooldown)
            m = compute_metrics(eq, tr)
            return m['calmar'] if np.isfinite(m['calmar']) else -999
        except Exception:
            return -999


def walk_forward(df, train_months=1, test_weeks=1, n_trials=150):

    results   = []
    all_equity = []
    params_log = []

    start_date = df['Datetime'].min()
    end_date   = df['Datetime'].max()

    window_start = start_date
    total_time   = 0

    while True:
        train_end = window_start + pd.DateOffset(months=train_months)
        test_end  = train_end   + pd.DateOffset(weeks=test_weeks)

        if test_end > end_date:
            break

        df_train = df[(df['Datetime'] >= window_start) & (df['Datetime'] < train_end)].copy()
        df_test  = df[(df['Datetime'] >= train_end)    & (df['Datetime'] < test_end)].copy()

        if len(df_train) < 500 or len(df_test) < 100:
            window_start += pd.DateOffset(weeks=test_weeks)
            continue

        # training
        t0    = time.time()
        study = optuna.create_study(direction='maximize')
        study.optimize(Objective(df_train), n_trials=n_trials, n_jobs=-1)
        total_time += time.time() - t0
        best = study.best_params

        # test
        sig    = generate_signals(df_test, rsi_ob=best['rsi_ob'], rsi_os=best['rsi_os'])
        eq, tr = run_backtest(sig,
                              atr_mult=best['atr_mult'],
                              take_profit_mult=best['take_profit_mult'],
                              risk_pct=best['risk_pct'],
                              cooldown=best['cooldown'])
        m = compute_metrics(eq, tr)
        m['window_start'] = window_start.strftime('%Y-%m-%d')
        m['window_end']   = test_end.strftime('%Y-%m-%d')
        m['best_params']  = best

        results.append(m)
        params_log.append(best)

        all_equity.append(eq)

        print(f"  {m['window_start']} -> {m['window_end']} | "
              f"Calmar: {m['calmar']:>7.4f} | "
              f"Sharpe: {m['sharpe']:>7.4f} | "
              f"WinRate: {m['win_rate']:.2%} | "
              f"Trades: {m['total_trades']}")

        window_start += pd.DateOffset(weeks=test_weeks)

    print(f"\nTotal optimization time: {total_time/60:.1f} minutes")

    import json

    results_df = pd.DataFrame([{
        'window_start': r['window_start'],
        'window_end': r['window_end'],
        'calmar': r['calmar'],
        'sharpe': r['sharpe'],
        'sortino': r['sortino'],
        'max_drawdown': r['max_drawdown'],
        'win_rate': r['win_rate'],
        'ann_return': r['ann_return'],
        'total_trades': r['total_trades']
    } for r in results])
    results_df.to_csv('results/walk_forward_results.csv', index=False)

    with open('results/best_params_log.json', 'w') as f:
        json.dump(params_log, f, indent=2)

    return results, params_log