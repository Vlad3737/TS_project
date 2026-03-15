import pandas as pd
from src.data_loader import download_m4_hourly
from src.utils import create_validation_windows, calculate_metrics

# Загружаем данные
df = pd.read_csv("data/m4_hourly.csv")
df['ds'] = pd.to_datetime(df['ds'])

print(f"Всего уникальных рядов: {df['unique_id'].nunique()}")
print(f"Диапазон дат: {df['ds'].min()} - {df['ds'].max()}")

# Создаем окна валидации
windows = create_validation_windows(
    df, 
    n_windows=4, 
    horizon=48,
    min_train_length=7*24,
    step=24
)

# Проверим первое окно первого ряда
first_id = df['unique_id'].iloc[0]
first_series = df[df['unique_id'] == first_id].reset_index(drop=True)
train_idx, test_idx = windows[first_id][0]

print(f"\nПример для ряда {first_id}:")
print(f"Длина всего ряда: {len(first_series)}")
print(f"Размер обучения: {len(train_idx)} точек (около {len(train_idx)/24:.1f} дней)")
print(f"Размер теста: {len(test_idx)} точек ({len(test_idx)} часов)")
print(f"Тестовый период: с {first_series.loc[test_idx[0], 'ds']} по {first_series.loc[test_idx[-1], 'ds']}")