
import numpy as np
import pandas as pd

def generate_signals(df, rsi_ob=60, rsi_os=40, macd_thresh=0):
    df = df.copy()

    # lon term
    trend_up   = df['ema200'] > df['ema500']
    trend_down = df['ema200'] < df['ema500']

    # macd
    macd_cross_up   = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    macd_cross_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

    # rsi
    rsi_ok_long  = df['rsi'] < rsi_ob
    rsi_ok_short = df['rsi'] > rsi_os

    # bollinger
    bb_ok_long  = df['Close'] < df['bb_mid']
    bb_ok_short = df['Close'] > df['bb_mid']

    # entry macd
    long_cond  = macd_cross_up   & trend_up   & (rsi_ok_long  | bb_ok_long)
    short_cond = macd_cross_down & trend_down & (rsi_ok_short | bb_ok_short)

    df['signal'] = 0
    df.loc[long_cond,  'signal'] =  1
    df.loc[short_cond, 'signal'] = -1

    return df

def run_backtest(df, atr_mult=3.0, take_profit_mult=2.0, risk_pct=0.01, fee=0.00125, cooldown=10):
    capital        = 10_000.0
    equity         = capital
    position       = 0
    entry_price    = 0.0
    stop_loss      = 0.0
    take_profit    = 0.0
    units          = 0.0
    cooldown_count = 0

    equity_curve = []
    trades       = []

    for i in range(len(df)):
        row    = df.iloc[i]
        price  = row['Close']
        signal = row['signal']
        atr    = row['atr']

        # close position
        if i > 0:
            time_diff = (row['Datetime'] - df.iloc[i-1]['Datetime']).total_seconds() / 60
            if time_diff > 60 and position != 0:
                if position == 1:
                    pnl    = (price - entry_price) * units
                    cost   = price * units * fee
                    equity += pnl - cost
                    trades.append({'pnl': pnl - cost, 'exit': price, 'type': 'long'})
                elif position == -1:
                    pnl    = (entry_price - price) * units
                    cost   = price * units * fee
                    equity += pnl - cost
                    trades.append({'pnl': pnl - cost, 'exit': price, 'type': 'short'})
                position       = 0
                cooldown_count = 0

        if cooldown_count > 0:
            cooldown_count -= 1

        # exits
        if position == 1:
            if price <= stop_loss or price >= take_profit or signal == -1:
                pnl    = (price - entry_price) * units
                cost   = price * units * fee
                equity += pnl - cost
                trades.append({'pnl': pnl - cost, 'exit': price, 'type': 'long'})
                position       = 0
                cooldown_count = cooldown

        elif position == -1:
            if price >= stop_loss or price <= take_profit or signal == 1:
                pnl    = (entry_price - price) * units
                cost   = price * units * fee
                equity += pnl - cost
                trades.append({'pnl': pnl - cost, 'exit': price, 'type': 'short'})
                position       = 0
                cooldown_count = cooldown

        # new position
        if position == 0 and signal != 0 and cooldown_count == 0:
            stop_dist = atr * atr_mult
            if stop_dist > 0:
                units     = (equity * risk_pct) / stop_dist
                max_units = (equity * 0.95) / price
                units     = min(units, max_units)
                cost      = price * units * fee

                if signal == 1:
                    entry_price = price
                    stop_loss   = price - stop_dist
                    take_profit = price + stop_dist * take_profit_mult
                    position    = 1

                elif signal == -1:
                    entry_price = price
                    stop_loss   = price + stop_dist
                    take_profit = price - stop_dist * take_profit_mult
                    position    = -1

                equity -= cost

        equity_curve.append(max(equity, 0))

    return np.array(equity_curve), trades

def compute_metrics(equity_curve, trades, freq=105120):

    equity  = np.array(equity_curve)
    returns = np.diff(equity) / equity[:-1]

    if len(returns) == 0 or returns.std() == 0:
        return {'sharpe': 0, 'sortino': 0, 'calmar': 0, 'max_drawdown': 0, 'win_rate': 0}

    ann_return  = (equity[-1] / equity[0]) ** (freq / len(returns)) - 1
    sharpe      = (returns.mean() / returns.std()) * np.sqrt(freq)

    downside    = returns[returns < 0]
    sortino     = (returns.mean() / downside.std()) * np.sqrt(freq) if len(downside) > 0 else 0

    peak        = np.maximum.accumulate(equity)
    drawdown    = (equity - peak) / peak
    max_dd      = drawdown.min()

    calmar      = ann_return / abs(max_dd) if max_dd != 0 else 0

    wins        = [t for t in trades if t['pnl'] > 0]
    win_rate    = len(wins) / len(trades) if trades else 0

    return {
        'sharpe':       round(sharpe, 4),
        'sortino':      round(sortino, 4),
        'calmar':       round(calmar, 4),
        'max_drawdown': round(max_dd, 4),
        'win_rate':     round(win_rate, 4),
        'ann_return':   round(ann_return, 4),
        'total_trades': len(trades)
    }