import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

class NaiveForecast:
    """Наивный прогноз: последнее значение повторяется на весь горизонт"""
    
    def __init__(self, horizon: int = 48):
        self.horizon = horizon
        self.last_value = None
        
    def fit(self, series: np.ndarray):
        """Запоминаем последнее значение ряда"""
        self.last_value = series[-1]
        return self
    
    def predict(self) -> np.ndarray:
        """Возвращаем массив из последнего значения"""
        return np.full(self.horizon, self.last_value)


class SeasonalNaiveForecast:
    """Сезонный наивный прогноз: значение из предыдущего сезона"""
    
    def __init__(self, horizon: int = 48, seasonality: int = 24):
        self.horizon = horizon
        self.seasonality = seasonality
        
    def fit(self, series: np.ndarray):
        """Запоминаем последний сезон"""
        self.last_season = series[-self.seasonality:]
        return self
    
    def predict(self) -> np.ndarray:
        """Повторяем последний сезон, чтобы покрыть горизонт"""
        n_repeats = (self.horizon // self.seasonality) + 1
        repeated = np.tile(self.last_season, n_repeats)
        return repeated[:self.horizon]


def evaluate_baseline(
    df: pd.DataFrame,
    windows_by_id: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    baseline_name: str,
    horizon: int = 48,
    seasonality: int = 24
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Оценивает бейзлайн на всех окнах.
    
    Returns:
        predictions_by_window: словарь {номер_окна: массив_предсказаний}
        targets_by_window: словарь {номер_окна: массив_истинных_значений}
    """
    predictions_by_window = {i: [] for i in range(len(next(iter(windows_by_id.values()))))}
    targets_by_window = {i: [] for i in range(len(next(iter(windows_by_id.values()))))}
    
    for unique_id in df['unique_id'].unique():
        series = df[df['unique_id'] == unique_id]['y'].values
        windows = windows_by_id[unique_id]
        
        for window_idx, (train_idx, test_idx) in enumerate(windows):
            # Обучающая и тестовая выборка
            train = series[train_idx]
            test = series[test_idx]
            
            # Выбираем модель
            if baseline_name == 'naive':
                model = NaiveForecast(horizon=horizon)
            elif baseline_name == 'seasonal_naive':
                model = SeasonalNaiveForecast(horizon=horizon, seasonality=seasonality)
            else:
                raise ValueError(f"Unknown baseline: {baseline_name}")
            
            # Предсказание
            model.fit(train)
            pred = model.predict()
            
            # Сохраняем
            predictions_by_window[window_idx].extend(pred)
            targets_by_window[window_idx].extend(test)
    
    # Преобразуем в numpy массивы
    for window_idx in predictions_by_window:
        predictions_by_window[window_idx] = np.array(predictions_by_window[window_idx])
        targets_by_window[window_idx] = np.array(targets_by_window[window_idx])
    
    return predictions_by_window, targets_by_window