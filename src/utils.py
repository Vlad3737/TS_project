import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import TimeSeriesSplit

def create_validation_windows(
    df: pd.DataFrame, 
    n_windows: int = 4, 
    horizon: int = 48,
    min_train_length: int = 7*24,
    step: int = 24
) -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    """
    Создает индексы для кросс-валидации с несколькими окнами.
    
    Возвращает словарь: для каждого unique_id список кортежей (train_idx, test_idx)
    """
    windows_by_id = {}
    
    for unique_id in df['unique_id'].unique():
        series = df[df['unique_id'] == unique_id].reset_index(drop=True)
        n = len(series)
        
        windows = []
        # Начинаем с минимальной длины и двигаемся вперед
        for i in range(n_windows):
            train_end = min_train_length + i * step
            test_start = train_end
            test_end = test_start + horizon
            
            # Проверяем, что тестовая выборка помещается в данные
            if test_end <= n:
                train_idx = np.arange(train_end)
                test_idx = np.arange(test_start, test_end)
                windows.append((train_idx, test_idx))
        
        windows_by_id[unique_id] = windows
        print(f"ID {unique_id}: создано {len(windows)} окон")
    
    return windows_by_id

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет метрики качества.
    Для M4 обычно используют SMAPE и MASE.
    """
    # SMAPE (Symmetric Mean Absolute Percentage Error)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    smape = np.mean(2 * np.abs(y_true - y_pred) / denominator) * 100
    
    # MAE (Mean Absolute Error)
    mae = np.mean(np.abs(y_true - y_pred))
    
    # RMSE (Root Mean Square Error)
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    
    return {
        'smape': smape,
        'mae': mae,
        'rmse': rmse
    }

def compute_metrics_across_windows(
    predictions: Dict[int, np.ndarray],  # окно -> предсказания для всех рядов
    targets: Dict[int, np.ndarray],      # окно -> истинные значения
    window_ids: List[int]
) -> pd.DataFrame:
    """
    Считает метрики для каждого окна и возвращает DataFrame.
    """
    results = []
    for window in window_ids:
        y_true = targets[window]
        y_pred = predictions[window]
        
        # Считаем метрики для этого окна
        metrics = calculate_metrics(y_true, y_pred)
        metrics['window'] = window
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    # Добавляем статистики по окнам
    summary = {
        'metric': ['mean_smape', 'std_smape', 'mean_mae', 'std_mae'],
        'value': [
            df_results['smape'].mean(),
            df_results['smape'].std(),
            df_results['mae'].mean(),
            df_results['mae'].std()
        ]
    }
    
    return df_results, pd.DataFrame(summary)