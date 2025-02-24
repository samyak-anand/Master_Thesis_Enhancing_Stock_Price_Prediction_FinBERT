# Import necessary libraries
import yfinance as yf
import pandas as pd
from datetime import datetime

# =========================
# Define List of Stock Tickers
# =========================
tickers = [
    'TSLA', 'AAPL', 'TSM', 'AMZN', 'MSFT', 'PG', 'AMD', 'META', 'NIO', 'NFLX',
    'GOOG', 'PYPL', 'DIS', 'COST', 'KO', 'INTC', 'ENPH', 'BA', 'CRM', 'ZS',
    'XPEV', 'VZ', 'BX', 'F', 'NOC'
]

# =========================
# Function to Bulk Download Stock Data
# =========================
def bulk_download_stock_data(ticker_list, start_date, end_date):
    """
    Downloads historical stock data for a list of tickers using Yahoo Finance.

    Parameters:
        ticker_list (list): List of stock tickers.
        start_date (str): Start date for data retrieval (YYYY-MM-DD).
        end_date (str): End date for data retrieval (YYYY-MM-DD).

    Returns:
        pd.DataFrame: Combined stock data for all tickers.
    """

    print("Downloading bulk data for all tickers...")
    
    # Download all tickers' historical data at once
    bulk_data = yf.download(ticker_list, start=start_date, end=end_date, group_by='ticker')

    # List to store processed stock data
    all_data = []

    for ticker in ticker_list:
        print(f"Processing data for {ticker}...")
        try:
            # Extract individual ticker data
            if (ticker,) in bulk_data.columns:
                stock_data = bulk_data[ticker].copy()
            else:
                print(f"No data found for {ticker}")
                continue

            # Fetch additional financial metrics from Yahoo Finance
            stock_info = yf.Ticker(ticker).info
            stock_data['Ticker'] = ticker
            stock_data['Market Cap'] = stock_info.get('marketCap', None)
            stock_data['P/E Ratio'] = stock_info.get('trailingPE', None)
            stock_data['Dividend Yield'] = stock_info.get('dividendYield', None)
            stock_data['EPS'] = stock_info.get('trailingEps', None)

            # Reset index to bring Date from index to a column
            stock_data.reset_index(inplace=True)

            # Append processed data to the list
            all_data.append(stock_data)

        except Exception as e:
            print(f"Error processing data for {ticker}: {e}")

    # Combine all individual stock data into a single DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)

    # Select relevant columns
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
                        'Ticker', 'Market Cap', 'P/E Ratio', 'Dividend Yield', 'EPS']
    
    return combined_data[required_columns]

# =========================
# Define Date Range and Download Data
# =========================
start_date = '2020-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')

# Download stock data
stock_data = bulk_download_stock_data(tickers, start_date, end_date)

# =========================
# Data Preview and Validation
# =========================

# Preview the data
print("Preview of downloaded stock data:")
print(stock_data.head())

# Save to CSV
output_file = 'historical_stock_data.csv'
stock_data.to_csv(output_file, index=False)
print(f"Stock data saved to '{output_file}'")

# Check for missing data
print("\nMissing Data in Stock DataFrame:")
print(stock_data.isnull().sum())

# Check total count of each column
print("\nTotal Data Count per Column:")
print(stock_data.count())
