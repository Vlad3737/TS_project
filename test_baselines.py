import pandas as pd
import numpy as np
from src.utils import create_validation_windows, calculate_metrics, compute_metrics_across_windows
from src.baselines import evaluate_baseline
import matplotlib.pyplot as plt

# Загружаем данные
df = pd.read_csv("data/m4_hourly.csv")
print("Данные загружены")

# Создаем окна валидации
windows_by_id = create_validation_windows(
    df, 
    n_windows=4, 
    horizon=48,
    min_train_length=7*24,
    step=24
)

# Оцениваем Naive
print("\n" + "="*50)
print("Оценка Naive модели")
naive_preds, naive_targets = evaluate_baseline(
    df, windows_by_id, 
    baseline_name='naive',
    horizon=48
)

naive_results, naive_summary = compute_metrics_across_windows(
    naive_preds, naive_targets, 
    window_ids=[0, 1, 2, 3]
)

print("\nРезультаты по окнам:")
print(naive_results)
print("\nСводка:")
print(naive_summary)

# Оцениваем SeasonalNaive
print("\n" + "="*50)
print("Оценка SeasonalNaive модели")
seasonal_preds, seasonal_targets = evaluate_baseline(
    df, windows_by_id, 
    baseline_name='seasonal_naive',
    horizon=48,
    seasonality=24
)

seasonal_results, seasonal_summary = compute_metrics_across_windows(
    seasonal_preds, seasonal_targets, 
    window_ids=[0, 1, 2, 3]
)

print("\nРезультаты по окнам:")
print(seasonal_results)
print("\nСводка:")
print(seasonal_summary)

# Визуализация для первого ряда, первого окна
first_id = df['unique_id'].iloc[0]
first_series = df[df['unique_id'] == first_id].reset_index(drop=True)
train_idx, test_idx = windows_by_id[first_id][0]

train = first_series.iloc[train_idx]['y'].values
test = first_series.iloc[test_idx]['y'].values

# Обучаем модели на первом ряду
from src.baselines import NaiveForecast, SeasonalNaiveForecast

naive = NaiveForecast(horizon=48)
naive.fit(train)
naive_pred = naive.predict()

seasonal = SeasonalNaiveForecast(horizon=48, seasonality=24)
seasonal.fit(train)
seasonal_pred = seasonal.predict()

# Рисуем
plt.figure(figsize=(12, 6))
plt.plot(range(len(train)), train, label='Train', color='blue', alpha=0.5)
plt.plot(range(len(train), len(train)+len(test)), test, label='Actual Test', color='green', linewidth=2)
plt.plot(range(len(train), len(train)+len(test)), naive_pred, '--', label='Naive', color='red')
plt.plot(range(len(train), len(train)+len(test)), seasonal_pred, '--', label='Seasonal Naive', color='orange')
plt.axvline(x=len(train), color='black', linestyle=':', alpha=0.7)
plt.title(f'Бейзлайны для ряда {first_id}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('results/baselines_example.png')
plt.show()