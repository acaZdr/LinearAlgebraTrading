import pandas as pd
import numpy as np
from typing import Tuple, List


def read_data(file_path: str) -> pd.DataFrame:
    try:
        data = pd.read_csv(file_path)
        return pd.DataFrame(data)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        print(f"Error: The file at {file_path} is empty")
        return pd.DataFrame()
    except pd.errors.ParserError:
        print(f"Error: Unable to parse the file at {file_path}")
        return pd.DataFrame()


def calculate_metrics(returns: List[float]) -> Tuple[float, float, float]:
    returns = pd.Series(returns)
    sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    total_return = (1 + returns).prod() - 1
    return sharpe_ratio, max_drawdown, total_return


def evaluate_strategy(df: pd.DataFrame, initial_investment: float = 1000) -> Tuple[float, float, pd.DataFrame]:
    holding = False
    entry_price = 0
    capital = initial_investment
    trades = []
    returns = []

    for index, row in df.iterrows():
        if row["Predicted"] == 2 and not holding:
            entry_price = row["AbsolutePrice"]
            entry_index = index
            holding = True
        elif row["Predicted"] == 0 and holding:
            exit_price = row["AbsolutePrice"]
            exit_index = index
            trade_return = (exit_price - entry_price) / entry_price
            capital_before = capital
            capital *= (1 + trade_return)
            relative_profit = (capital - capital_before) / capital_before
            holding = False
            trades.append({
                "entry_index": entry_index,
                "exit_index": exit_index,
                "entry_price": round(entry_price, 6),
                "exit_price": round(exit_price, 6),
                "trade_return": round(trade_return, 6),
                "capital_after_trade": round(capital, 2),
                "relative_profit": round(relative_profit, 6)
            })
            returns.append(trade_return)

    if holding:
        exit_price = df.iloc[-1]["AbsolutePrice"]
        exit_index = df.index[-1]
        trade_return = (exit_price - entry_price) / entry_price
        capital_before = capital
        capital *= (1 + trade_return)
        relative_profit = (capital - capital_before) / capital_before
        trades.append({
            "entry_index": entry_index,
            "exit_index": exit_index,
            "entry_price": round(entry_price, 6),
            "exit_price": round(exit_price, 6),
            "trade_return": round(trade_return, 6),
            "capital_after_trade": round(capital, 2),
            "relative_profit": round(relative_profit, 6)
        })
        returns.append(trade_return)

    trades_df = pd.DataFrame(trades)
    return capital, (capital - initial_investment) / initial_investment, trades_df


def calculate_monthly_profits(df: pd.DataFrame, intervals_per_month: int) -> pd.DataFrame:
    monthly_profits = []
    hodl_profits = []

    total_intervals = len(df)
    num_months = total_intervals // intervals_per_month

    for month in range(num_months):
        start_idx = month * intervals_per_month
        end_idx = start_idx + intervals_per_month - 1

        month_data = df.iloc[start_idx:end_idx + 1]
        _, monthly_profit, _ = evaluate_strategy(month_data)

        start_price = month_data.iloc[0]["AbsolutePrice"]
        end_price = month_data.iloc[-1]["AbsolutePrice"]
        hodl_profit = (end_price / start_price - 1) * 100

        monthly_profits.append(monthly_profit * 100)  # Convert to percentage
        hodl_profits.append(hodl_profit)

    # Calculate overall HODL profit for the entire dataset
    overall_hodl_profit = (df.iloc[-1]["AbsolutePrice"] / df.iloc[0]["AbsolutePrice"] - 1) * 100

    results_df = pd.DataFrame({
        "Month": range(1, num_months + 1),
        "Strategy Profit (%)": monthly_profits,
        "HODL Profit (%)": hodl_profits
    })

    print("Monthly Strategy and HODL Profits:")
    print(results_df)

    print(f"Overall HODL Profit for the entire dataset: {overall_hodl_profit:.2f}%")

    return results_df


def main():
    file_path = "../results/outputs/test_pca_False_results_with_prices.csv"
    df = read_data(file_path)

    if df.empty:
        return

    intervals_per_month = len(df)// 17
    monthly_profits_df = calculate_monthly_profits(df, intervals_per_month)

    initial_investment = 1000
    final_capital, total_return, trades_df = evaluate_strategy(df, initial_investment)

    print(f"Initial Investment: ${initial_investment:.2f}")
    print(f"Final Capital: ${final_capital:.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Number of Trades: {len(trades_df)}")

    trades_df.to_csv("../results/outputs/trades_log.csv", index=False)


if __name__ == "__main__":
    main()
