# Enhancing Stock Price Prediction with Sentiment Analysis and Advanced Neural Networks

**A hybrid FinBERT + LSTM framework for sentiment-driven, explainable stock market forecasting.**

This project was built as part of an M.Sc. Data Science thesis (IU International University of Applied Sciences). It investigates whether fusing financial-domain sentiment — extracted from social media text using **FinBERT** — with structured market data measurably improves stock forecasting over price-only baselines, and whether the resulting models can be made genuinely explainable rather than left as black boxes.

📄 Full write-up: *[Can a Tweet Move a Stock? Inside a Hybrid FinBERT–LSTM Framework for Explainable Market Forecasting](https://medium.com/@samyak.anand/can-a-tweet-move-a-stock-inside-a-hybrid-finbert-lstm-framework-for-explainable-market-forecasting-b40a57104cad)* (Medium)
🎓 Thesis: *Enhancing Stock Price Prediction with Sentiment Analysis and Advanced Neural Networks* — Samyak Anand, Supervisor: Dr. Stefan Stöckl

---

## 📌 Objective

Build and benchmark a multimodal forecasting framework that:

- Extracts financial sentiment from social media text using **FinBERT** (domain-adapted BERT)
- Fuses sentiment scores with structured OHLCV market data (price, volume, technical indicators)
- Feeds the fused feature set into an **LSTM** to model temporal dependencies between sentiment and price behavior
- Benchmarks classical ML baselines (Logistic Regression, SVM, Random Forest, XGBoost) against deep learning and transformer approaches (LSTM, BERT, FinBERT)
- Applies **SHAP** (SHapley Additive Explanations) to make every model's decisions interpretable, not just accurate

---

## 🧠 Architecture — a model tournament, not a single model

The framework is built in escalating tiers, where each model earns its place by addressing a specific limitation of the one before it:

| Tier | Model | Why it's there |
|---|---|---|
| Baseline | Logistic Regression | Cheap, interpretable, linear — validates the feature pipeline before adding complexity |
| Baseline | SVM (linear → RBF) | Tests whether sentiment classes are linearly separable, then whether a non-linear kernel does better |
| Baseline | Random Forest | Captures non-linear feature interactions; gives interpretable feature importances for free |
| Advanced | XGBoost | Gradient boosting for a tighter fit on the combined structured + sentiment feature space |
| Advanced | LSTM | The only model that explicitly captures *sequence* — how sentiment and price evolve together over time |
| Advanced | BERT | General-purpose transformer baseline, used to isolate how much benefit comes from *financial* domain pretraining |
| Advanced | FinBERT | Domain-adapted transformer at the center of the framework — pretrained specifically on financial text |

---

## 📂 Data

Two datasets, temporally merged on date and ticker:

- **Market data**: ~32,000 daily trading records (via `yfinance`), covering high-profile tickers (Tesla, Apple, Amazon, and others) — open/high/low/close prices, volume, market cap, P/E ratio, EPS, dividend yield.
- **Sentiment data**: ~80,000 stock-related tweets/social posts, filtered for relevance to ~24 technology and consumer-sector companies, labeled Positive / Neutral / Negative via FinBERT.

**Class distribution note:** the sentiment labels are heavily skewed toward Neutral (~72% of the dataset). This is why class-balancing (SMOTE, applied strictly to the training split only, to avoid leaking synthetic examples into validation/test) was a required preprocessing step, not an optional one — and it's worth keeping in mind when reading any accuracy number below: a model that predicts "Neutral" every time already scores ~72% on this data.

---

## 🛠 Tools & Technologies

- **Language**: Python 3.10+
- **NLP / Sentiment**: `transformers` (Hugging Face) — FinBERT, BERT
- **Text processing**: `nltk`, `re`
- **ML models**: `scikit-learn`, `xgboost`, `imbalanced-learn` (SMOTE)
- **Deep learning**: TensorFlow / Keras (LSTM)
- **Explainability**: `shap`
- **Data handling**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `wordcloud`, `mplfinance`
- **Utilities**: `tqdm`

---

## 🧩 Project Structure

```
├── data/                # Stock market data + financial sentiment/tweet datasets
├── notebooks/            # Jupyter notebooks for experimentation
├── models/               # Saved trained model artifacts
├── ST_NLP.ipynb           # Main notebook: sentiment extraction + ML/DL modeling
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🔍 Key Components

**1. Data Preprocessing**
Cleaning and formatting stock and sentiment data; tokenization, lemmatization, stopword removal; handling missing values and class imbalance (SMOTE, train-fold only).

**2. Sentiment Analysis with FinBERT**
Hugging Face `transformers` pipeline scores each post as Positive / Neutral / Negative, producing both a categorical label and a sentiment score used as a downstream feature.

**3. Classical ML Benchmarks**
Logistic Regression, Random Forest, SVM (RBF), and XGBoost trained on text-representation features (Bag-of-Words, TF-IDF) and later on the fused multimodal feature set, to establish a baseline before investing in deep learning.

**4. LSTM for Temporal Forecasting**
Combines historical price data with sentiment signals over time to capture temporal dependencies that the tree-based and linear models can't model natively.

**5. Explainability**
SHAP and native feature importance applied to the XGBoost classifier to interpret which features — market indicators, sentiment scores — drive individual predictions, not just aggregate accuracy.

---

## 📊 Results

The table below reflects the final, consolidated comparison across all models (see thesis Table 27 for the canonical source).

| Model | Feature Set | Accuracy | F1-Score | RMSE | MAE |
|---|---|---|---|---|---|
| Logistic Regression | Multimodal | 55.1% | 55% | – | – |
| Random Forest | Multimodal | 91.6% | 92% | – | – |
| XGBoost | Multimodal | 78.7% | 79% | – | – |
| LSTM | Sequential multimodal | – | – | 8.23 | 6.92 |
| BERT (sentiment) | Transformer | 92.0% | 92% | – | – |
| FinBERT (sentiment) | Transformer | 99.9% | 99% | – | – |

**Read this table carefully, not as a leaderboard.** A few things worth knowing before citing any number here:

- The ~72% Neutral-class prevalence means a model that predicts nothing and guesses "Neutral" every time already scores ~72% accuracy — several earlier-stage/tuned classical models in the full experiment log come close to or below that trivial baseline. This table is being tightened to include that baseline explicitly.
- FinBERT's 99.9% is almost certainly inflated by some circularity between how sentiment labels were generated and how FinBERT itself was evaluated against them — treat it as an upper bound to be re-validated, not a headline result.
- Different experiment passes across the underlying notebooks (text-only features vs. multimodal, pre- vs. post-tuning) currently report slightly different numbers for the same model. This repo predates proper experiment tracking — a cleanup pass (MLflow/W&B, canonical result tables per run config) is in progress; until then, treat this table as directionally correct rather than exact.

---

## ⚠️ Known Limitations

- **Source representativeness** — social-media sentiment captures retail-investor mood, not institutional or algorithmic flows.
- **Linguistic ambiguity** — sarcasm, abbreviations, and financial slang remain hard for any sentiment model.
- **No majority-class baseline reported (yet)** — being added to make relative model lift honest and checkable.
- **Random (not time-aware) train/test split** — a rolling-window/chronological split would better reflect real deployment conditions for time-series data.
- **Explainability coverage** — SHAP is currently applied to XGBoost only, not to the LSTM or FinBERT components.
- **No financial-outcome metrics yet** — accuracy/F1 measure classification quality, not trading value; Sharpe ratio, drawdown, and transaction-cost-aware backtesting are not yet included.
- **Daily granularity** — misses intraday sentiment shifts and lag effects.

---

## 🔭 Future Enhancements

- Real-time sentiment ingestion (streaming pipelines, e.g. Kafka) instead of static, batch-processed data
- Broader, multilingual, multi-sector data sources (earnings calls, Reddit, regulatory filings — not just one platform)
- Advanced architectures: Temporal Fusion Transformers / Transformer-LSTM hybrids
- Explainability extended beyond XGBoost to LSTM and FinBERT (attention inspection, gradient-based explainers)
- Financial-outcome evaluation: Sharpe ratio, drawdown, alpha, realistic backtests with transaction costs
- Robustness testing against adversarial/coordinated sentiment manipulation
- Deployment as a web dashboard (Streamlit or Flask)
- Experiment tracking and a model registry (MLflow / Weights & Biases) to keep every reported metric traceable to one exact run

---

## ⚙️ Setup

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
jupyter notebook ST_NLP.ipynb
```

---

## 🔗 Links

- LinkedIn: [linkedin.com/in/samyak-anand9](https://linkedin.com/in/samyak-anand9)
- GitHub: [github.com/samyak-anand](https://github.com/samyak-anand)

---

*This is a first academic pass at the problem, published transparently — including its rough edges — because that's more useful to anyone building on it than a polished-looking result that doesn't hold up under scrutiny. Issues, questions, and pull requests welcome.*
