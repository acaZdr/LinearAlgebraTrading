import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange


def import_data(csv_path: str, limit=None) -> pd.DataFrame:
	df = pd.read_csv(csv_path)
	if limit:
		df = df.iloc[:limit, :6]
	else:
		df = df.iloc[:, :6]
	df.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']

	df['date'] = pd.to_datetime(df['date'], unit='ms')
	return df


def calculate_indicators(csv_file: str) -> pd.DataFrame:
	df = import_data(csv_file)

	# Ensure the data is sorted by date
	df = df.sort_values(by='date')

	# Calculate SMA_10
	sma_10 = SMAIndicator(df['Close'], window=10)
	df['SMA_10'] = sma_10.sma_indicator()

	# Calculate SMA_50
	sma_50 = SMAIndicator(df['Close'], window=50)
	df['SMA_50'] = sma_50.sma_indicator()

	# Calculate EMA_20
	ema_20 = EMAIndicator(df['Close'], window=20)
	df['EMA_20'] = ema_20.ema_indicator()

	# Calculate RSI_14
	rsi_14 = RSIIndicator(df['Close'], window=14)
	df['RSI_14'] = rsi_14.rsi()

	# Calculate MACD
	macd = MACD(df['Close'])
	df['MACD'] = macd.macd()

	# Calculate Bollinger Bands
	bb = BollingerBands(df['Close'])
	df['BB_High'] = bb.bollinger_hband()
	df['BB_Low'] = bb.bollinger_lband()

	# Calculate ATR_14
	try:
		atr_14 = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
		df['ATR_14'] = atr_14.average_true_range()
	except IndexError as ie:
		df['ATR_14'] = pd.NA

	return df


def save_to_csv(df: pd.DataFrame, csv_path: str):
	new_order = ['date', 'Open', 'High', 'Low', 'Close', 'Volume', 'SMA_10', 'SMA_50', 'EMA_20', 'RSI_14', 'MACD',
				 'BB_High', 'BB_Low', 'ATR_14']
	df = df.reindex(columns=new_order)
	df.to_csv(csv_path, index=False, mode='a', header=False)