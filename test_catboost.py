import pandas as pd
import numpy as np
from src.utils import create_validation_windows, compute_metrics_across_windows
from src.models import CatBoostForecaster, evaluate_model
import time

# Загружаем данные
df = pd.read_csv("data/m4_hourly.csv")
df['ds'] = pd.to_datetime(df['ds'])
print("Данные загружены")

# Создаем окна валидации
windows_by_id = create_validation_windows(
    df, 
    n_windows=4, 
    horizon=48,
    min_train_length=7*24,
    step=24
)

# Оцениваем CatBoost
print("\n" + "="*50)
print("Оценка CatBoost модели")

start_time = time.time()

catboost_preds, catboost_targets = evaluate_model(
    df,
    windows_by_id,
    model_class=CatBoostForecaster,
    model_params={
        'lags': [1, 2, 3, 23, 24, 25, 47, 48],
        'cat_features': ['hour', 'dayofweek'],
        'model_params': {
            'iterations': 100,
            'learning_rate': 0.05,
            'depth': 5,
            'loss_function': 'MAE',
            'verbose': 50,
            'random_seed': 42
        }
    },
    horizon=48
)

elapsed_time = time.time() - start_time
print(f"Время выполнения: {elapsed_time:.2f} секунд")

# Считаем метрики
catboost_results, catboost_summary = compute_metrics_across_windows(
    catboost_preds, catboost_targets,
    window_ids=[0, 1, 2, 3]
)

print("\nРезультаты по окнам (CatBoost):")
print(catboost_results)
print("\nСводка (CatBoost):")
print(catboost_summary)

# Сравнение с бейзлайнами
print("\n" + "="*50)
print("СРАВНЕНИЕ ВСЕХ МОДЕЛЕЙ:")
print("-"*50)
print(f"{'Модель':<20} {'SMAPE':<10} {'std_SMAPE':<10} {'MAE':<10}")
print("-"*50)
print(f"{'Naive':<20} {84.74:<10.2f} {3.93:<10.2f} {1379.78:<10.2f}")
print(f"{'SeasonalNaive':<20} {28.25:<10.2f} {4.67:<10.2f} {311.36:<10.2f}")
print(f"{'CatBoost':<20} {catboost_summary.iloc[0]['value']:<10.2f} "
      f"{catboost_summary.iloc[1]['value']:<10.2f} "
      f"{catboost_summary.iloc[2]['value']:<10.2f}")