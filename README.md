Enhancing Stock Price Prediction using FinBERT and LSTM
This project is part of an ongoing master's thesis aimed at improving the prediction of stock price movements by integrating financial sentiment analysis and time-series modeling. The hybrid approach leverages FinBERT, a domain-specific NLP model, alongside LSTM (Long Short-Term Memory) neural networks, enabling sentiment-driven forecasting from financial news.

📌 Objective
The goal of this project is to build a hybrid stock market forecasting model that:

Extracts sentiment from financial news using FinBERT

Integrates sentiment scores into a LSTM model for time-series prediction

Benchmarks performance against classical ML models (e.g., Logistic Regression, Random Forest, SVM, XGBoost)

Utilizes Explainable AI (e.g., SHAP) for interpreting model predictions

🛠 Tools & Technologies
Languages: Python 3.10+

Libraries:

pandas, numpy – Data handling

nltk, re, transformers – Text processing & sentiment analysis

matplotlib, seaborn, wordcloud, mplfinance – Visualization

scikit-learn, xgboost, imbalanced-learn – Machine learning models

tqdm – Loop progress tracking

NLP Model: FinBERT via Hugging Face Transformers

Time Series Model: LSTM (via TensorFlow/Keras)

EDA: Word clouds, sentiment distribution, candlestick charts

Model Evaluation: Accuracy, F1-score, ROC-AUC, confusion matrix

🧠 Project Structure
bash
Copy
Edit
├── data/                   # Financial news and stock datasets
├── notebooks/              # Jupyter Notebooks for experimentation
├── models/                 # Saved trained models
├── ST_NLP.ipynb            # Main notebook: sentiment + ML modeling
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
📂 Key Components
1. Data Preprocessing
Cleaning and formatting stock news and price data

Tokenization, lemmatization, and stopword removal

Handling missing values and class imbalance (SMOTE)

2. Sentiment Analysis with FinBERT
Use of Hugging Face Transformers pipeline for sentiment scoring

Assigning positive, negative, or neutral tags to financial headlines

3. Machine Learning Models
Models trained on vectorized sentiment + metadata:

Logistic Regression

Random Forest Classifier

XGBoost

SVM (RBF Kernel)

4. LSTM for Stock Price Forecasting
Incorporating both time-series price data and sentiment signals

Captures temporal dependencies in market movements

5. Evaluation & Visualization
Metrics: accuracy, F1-score, ROC-AUC, confusion matrix

Visuals: word clouds, sentiment bar plots, ROC curves, stock candlestick charts

📊 Sample Output
Sentiment Word Cloud

Stock Price Candlestick Chart

Confusion Matrix of Classifiers

SHAP-based feature importance visualizations

🔍 Future Enhancements
Integration with real-time news APIs

Multilingual sentiment analysis using multilingual BERT

Deployment as a web dashboard using Streamlit or Flask

  
