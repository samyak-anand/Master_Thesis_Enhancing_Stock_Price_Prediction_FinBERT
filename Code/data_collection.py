# Importing necessary libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from textblob import TextBlob


# Function to collect historical stock data
def get_stock_data(tickers, start_date, end_date):
    """
    Fetch historical stock data for multiple tickers from Yahoo Finance.

    Args:
        tickers (list): List of stock ticker symbols (e.g., ['AAPL', 'MSFT']).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Stock price data for all tickers.
    """
    stock_data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')
    all_data = []
    for ticker in tickers:
        if ticker in stock_data.columns.levels[0]:
            df = stock_data[ticker].copy()
            df.reset_index(inplace=True)
            df['Ticker'] = ticker
            df['Date'] = pd.to_datetime(df['Date']).dt.date  # Convert to date format
            all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


# Fetch news headlines from uploaded datasets
def fetch_news_from_csv():
    """
    Fetch news headlines from provided CSV files.

    Returns:
        pd.DataFrame: Combined news data from all sources with Time, Headline, and Description.
    """
    # Load datasets
    reuters = pd.read_csv('/home/samyak/PycharmProjects/Master_Thesis_Enhancing_Stock_Price_Prediction_FinBERT/News_Dataset/reuters_headlines.csv')
    guardian = pd.read_csv('/home/samyak/PycharmProjects/Master_Thesis_Enhancing_Stock_Price_Prediction_FinBERT/News_Dataset/guardian_headlines.csv')
    #cnbc = pd.read_csv('/home/samyak/PycharmProjects/Master_Thesis_Enhancing_Stock_Price_Prediction_FinBERT/News_Dataset/cnbc_headlines.csv')

    print("Standardizing 'Time' column and handling missing values...")
    for dataset, name in zip([ guardian, reuters], ['Guardian', 'Reuters']):
        print(f"Processing dataset: {name}")
        if 'Time' in dataset.columns:
            dataset['Time'] = dataset['Time'].str.replace(r"ET.*", "", regex=True).str.strip()
        else:
            dataset['Time'] = None

    print("Combining datasets...")
    combined_news_data = pd.concat(
        [
            #cnbc[['Time', 'Headlines', 'Description']].assign(Source='CNBC'),
            guardian[['Time', 'Headlines']].assign(Description=None, Source='Guardian'),
            reuters[['Time', 'Headlines', 'Description']].assign(Source='Reuters')
        ],
        ignore_index=True
    )

    print("Dropping rows with missing 'Headlines'...")
    combined_news_data.dropna(subset=['Headlines'], inplace=True)
    combined_news_data.reset_index(drop=True, inplace=True)
    combined_news_data['Time'] = pd.to_datetime(combined_news_data['Time'], errors='coerce')
    combined_news_data.sort_values(by='Time', inplace=True)
    combined_news_data.reset_index(drop=True, inplace=True)
    print(f"Final combined news dataset contains {len(combined_news_data)} rows.")
    return combined_news_data



# Analysis and visualization functions
def analyze_and_visualize_stock(stock_data):
    """
    Perform analysis and visualization for stock data.
    """
    print("\nStock Data Overview:")
    print(stock_data.describe())

    # Plot Closing Prices for Each Ticker
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])
    for ticker in stock_data['Ticker'].unique():
        ticker_data = stock_data[stock_data['Ticker'] == ticker]
        plt.figure(figsize=(10, 6))
        plt.plot(ticker_data['Date'], ticker_data['Close'], label=f"{ticker} Closing Price")
        plt.title(f"{ticker} Closing Prices Over Time")
        plt.xlabel("Date")
        plt.ylabel("Closing Price")
        plt.legend()
        plt.grid()
        plt.show()

def analyze_and_visualize_news(news_data):
    """
    Perform analysis and visualization for news data.
    """
    # Sentiment Analysis
    print("\nPerforming sentiment analysis on news headlines...")
    news_data['Sentiment'] = news_data['Headlines'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Average Sentiment by Source
    sentiment_by_source = news_data.groupby('Source')['Sentiment'].mean()
    print("\nAverage Sentiment by Source:")
    print(sentiment_by_source)

    # Visualize Sentiment by Source
    plt.figure(figsize=(10, 6))
    sentiment_by_source.plot(kind='bar', color=['blue', 'green', 'orange'])
    plt.title("Average Sentiment Polarity by Source")
    plt.xlabel("Source")
    plt.ylabel("Average Sentiment Polarity")
    plt.grid()
    plt.show()

    
# Main workflow
if __name__ == "__main__":
    print("Starting main workflow...")
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Add more tickers as needed
    start_date = '2018-03-20'  # Start date for fetching data
    end_date = '2020-07-18'  # End date for fetching data

    # Fetch stock data for multiple tickers
    try:
        stock_data = get_stock_data(tickers, start_date=start_date, end_date=end_date)
        print(f"Stock data fetched successfully. Shape: {stock_data.shape}")
    except Exception as e:
        print(f"Error fetching stock data: {e}")

    # Fetch news data from uploaded datasets
    try:
        news_data = fetch_news_from_csv()
        print(f"News data fetched successfully. Shape: {news_data.shape}")
    except ValueError as e:
        print(f"Error fetching news data: {e}")
        news_data = pd.DataFrame()

    # Perform analysis and visualization
    if not stock_data.empty:
        analyze_and_visualize_stock(stock_data)

    if not news_data.empty:
        analyze_and_visualize_news(news_data)

    # Save or display the result
    if not news_data.empty:
        print("Saving news data to 'news_data_combined.csv'...")
        news_data.to_csv("news_data_combined.csv", index=False)
        print("News data saved successfully.")
    else:
        print("No valid news data available.")

    print("Main workflow completed.")

    print("News Data", news_data)
    print("Stock Data", stock_data)