import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils import compute_metrics_across_windows
from src.baselines import evaluate_baseline
from src.models import CatBoostForecaster, evaluate_model
from src.ensemble import SimpleEnsemble, StackingEnsemble, evaluate_ensemble

df = pd.read_csv("data/m4_hourly.csv")
df['ds'] = pd.to_datetime(df['ds'])

# Загружаем windows (нужно сохранить из предыдущего запуска)
# Для простоты пересоздадим
from src.utils import create_validation_windows
windows_by_id = create_validation_windows(df, n_windows=4, horizon=48)

print("="*60)
print("ЭКСПЕРИМЕНТЫ С АНСАМБЛЯМИ")
print("="*60)

print("\n1. Собираем предсказания базовых моделей...")

# SeasonalNaive
seasonal_preds, seasonal_targets = evaluate_baseline(
    df, windows_by_id, 'seasonal_naive', horizon=48, seasonality=24
)

catboost_preds, catboost_targets = evaluate_model(
    df, windows_by_id,
    model_class=CatBoostForecaster,
    model_params={
        'lags': [1, 2, 3, 23, 24, 25, 47, 48],
        'cat_features': ['hour', 'dayofweek'],
        'model_params': {
            'iterations': 100,
            'learning_rate': 0.05,
            'depth': 5,
            'loss_function': 'MAE',
            'verbose': True,
            'random_seed': 42
        }
    },
    horizon=48
)

#Создаем словарь с предсказаниями всех моделей
all_predictions = {
    'seasonal_naive': seasonal_preds,
    'catboost': catboost_preds
}

# Простой ансамбль (равные веса)
print("\n2. Оцениваем SimpleEnsemble (равные веса)...")
simple_ensemble = SimpleEnsemble(all_predictions)
ensemble_preds_equal = simple_ensemble.predict()

ensemble_results_equal, ensemble_summary_equal = evaluate_ensemble(
    ensemble_preds_equal, 
    seasonal_targets,
    window_ids=[0, 1, 2, 3],
    ensemble_name='SimpleEnsemble (equal)'
)

print("\nРезультаты SimpleEnsemble:")
print(ensemble_summary_equal)

# Взвешенный ансамбль (оптимизируем веса)
print("\n3. Поиск оптимальных весов...")

# Перебираем веса для двух моделей
best_weights = None
best_smape = float('inf')
best_std = float('inf')

# Создаем сетку весов
for w1 in np.linspace(0, 1, 11):
    w2 = 1 - w1
    weights = [w1, w2]
    
    ensemble_preds = simple_ensemble.predict(weights=weights)
    _, summary = evaluate_ensemble(
        ensemble_preds, seasonal_targets,
        window_ids=[0, 1, 2, 3],
        ensemble_name='temp'
    )
    
    # Критерий: минимизируем комбинацию среднего и std
    score = summary.iloc[0]['mean_smape'] + summary.iloc[0]['std_smape']
    
    if score < best_smape + best_std:
        best_smape = summary.iloc[0]['mean_smape']
        best_std = summary.iloc[0]['std_smape']
        best_weights = [w1, w2]

print(f"Лучшие веса: SeasonalNaive={best_weights[0]:.2f}, CatBoost={best_weights[1]:.2f}")

ensemble_preds_weighted = simple_ensemble.predict(weights=best_weights)
ensemble_results_weighted, ensemble_summary_weighted = evaluate_ensemble(
    ensemble_preds_weighted, seasonal_targets,
    window_ids=[0, 1, 2, 3],
    ensemble_name='SimpleEnsemble (weighted)'
)

# Стекинг
print("\n4. Обучаем StackingEnsemble...")

# Используем окна 0,1 для обучения, окна 2,3 для теста
stacking = StackingEnsemble(
    base_predictions=all_predictions,
    targets=seasonal_targets,
    window_ids=[0, 1, 2, 3]
)

stacking.fit(val_window_ids=[0, 1])
stacking_preds = stacking.predict(test_window_ids=[2, 3])

# Оцениваем стекинг только на тестовых окнах
stacking_results, stacking_summary = evaluate_ensemble(
    stacking_preds, seasonal_targets,
    window_ids=[2, 3],
    ensemble_name='Stacking'
)

# 6. Сравнение всех моделей
print("\n" + "="*60)
print("ИТОГОВОЕ СРАВНЕНИЕ")
print("="*60)

# Собираем все результаты
all_results = pd.DataFrame([
    {'model': 'SeasonalNaive', 'mean_smape': 28.25, 'std_smape': 4.67},
    {'model': 'CatBoost', 'mean_smape': 30.86, 'std_smape': 5.85},
    ensemble_summary_equal.iloc[0].to_dict(),
    ensemble_summary_weighted.iloc[0].to_dict()
])

# Добавляем стекинг (только для окон 2-3)
stacking_row = stacking_summary.iloc[0].to_dict()
stacking_row['model'] = 'Stacking (windows 2-3)'
all_results = pd.concat([all_results, pd.DataFrame([stacking_row])], ignore_index=True)

print("\nСводная таблица:")
print(all_results[['model', 'mean_smape', 'std_smape', 'mean_mae']].round(2))

# 7. Визуализация
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Средний SMAPE vs Стабильность
ax1 = axes[0]
for _, row in all_results.iterrows():
    ax1.scatter(row['mean_smape'], row['std_smape'], s=100, label=row['model'])
    ax1.annotate(row['model'], 
                 (row['mean_smape'], row['std_smape']),
                 xytext=(5, 5), textcoords='offset points')

ax1.set_xlabel('Средний SMAPE')
ax1.set_ylabel('Std SMAPE (нестабильность)')
ax1.set_title('Среднее качество vs Стабильность')
ax1.grid(True, alpha=0.3)
ax1.legend()

# График 2: SMAPE по окнам
ax2 = axes[1]
# Собираем данные по окнам для основных моделей
windows_data = []
for model_name, preds in [('SeasonalNaive', seasonal_preds), 
                          ('CatBoost', catboost_preds),
                          ('Ensemble(weighted)', ensemble_preds_weighted)]:
    for window in [0, 1, 2, 3]:
        if window in preds and window in seasonal_targets:
            from src.utils import calculate_metrics
            metrics = calculate_metrics(seasonal_targets[window], preds[window])
            windows_data.append({
                'model': model_name,
                'window': window,
                'smape': metrics['smape']
            })

windows_df = pd.DataFrame(windows_data)
sns.barplot(data=windows_df, x='window', y='smape', hue='model', ax=ax2)
ax2.set_title('SMAPE по окнам')
ax2.set_xlabel('Окно валидации')
ax2.set_ylabel('SMAPE')

plt.tight_layout()
plt.savefig('results/ensemble_comparison.png', dpi=150)
plt.show()