import pandas as pd
import matplotlib.pyplot as plt

# Create example performance data
data = {
    'Model': ['BoW + Logistic Regression', 'BoW + XGBoost', 'FinBERT + XGBoost', 'FinBERT + LSTM'],
    'Accuracy': [0.681, 0.783, 0.816, 0.834],
    'F1-Score': [0.66, 0.76, 0.79, 0.81]
}

df = pd.DataFrame(data)

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = range(len(df))

# Bars
bar1 = plt.bar(index, df['Accuracy'], bar_width, label='Accuracy')
bar2 = plt.bar([i + bar_width for i in index], df['F1-Score'], bar_width, label='F1-Score')

# Labels and formatting
plt.xlabel('Model')
plt.ylabel('Score')
plt.title('BoW vs FinBERT-based Model Performance')
plt.xticks([i + bar_width / 2 for i in index], df['Model'], rotation=15, ha='right')
plt.ylim(0.6, 0.9)
plt.legend()

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


