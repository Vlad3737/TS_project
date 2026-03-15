import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from typing import List, Tuple, Dict
from sklearn.base import BaseEstimator, RegressorMixin
from tqdm import tqdm


class CatBoostForecaster(BaseEstimator, RegressorMixin):
    """
    CatBoost для прогнозирования временных рядов с лаговыми признаками
    """
    def __init__(
        self, 
        horizon: int = 48,
        lags: List[int] = [1, 2, 3, 23, 24, 25, 47, 48],  # важные лаги: предыдущие часы и вчерашние
        cat_features: List[str] = ['hour', 'dayofweek'],
        model_params: dict = None
    ):
        self.horizon = horizon
        self.lags = lags
        self.cat_features = cat_features
        self.model_params = model_params or {
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'MAE',
            'verbose': False,
            'random_seed': 42
        }
        self.model = None
        
    def _create_features(self, series: np.ndarray, dates: pd.DatetimeIndex):
        """
        Создает признаки для обучения:
        - Лаговые значения
        - Временные признаки (час, день недели)
        """
        df = pd.DataFrame({'y': series, 'ds': dates})
        
        # Лаговые признаки
        for lag in self.lags:
            if lag < len(series):
                df[f'lag_{lag}'] = df['y'].shift(lag)
        
        # Временные признаки
        df['hour'] = df['ds'].dt.hour
        df['dayofweek'] = df['ds'].dt.dayofweek
        
        # Удаляем строки с NaN (из-за лагов)
        df = df.dropna().reset_index(drop=True)
        
        return df
    
    def fit(self, series: np.ndarray, dates: pd.DatetimeIndex):
        """
        Обучает модель на одном временном ряде
        """
        # Создаем признаки
        df = self._create_features(series, dates)
        
        # Определяем признаки для обучения
        feature_cols = [col for col in df.columns if col not in ['y', 'ds']]
        X = df[feature_cols]
        y = df['y']
        
        # Определяем индексы категориальных признаков
        cat_features_indices = [i for i, col in enumerate(feature_cols) if col in self.cat_features]
        
        # Обучаем CatBoost
        self.model = CatBoostRegressor(**self.model_params)
        self.model.fit(
            X, y, 
            cat_features=cat_features_indices,
            verbose=False
        )
        
        # Сохраняем последние значения для лагов при прогнозе
        self.last_values = series[-max(self.lags):]
        self.last_dates = dates[-max(self.lags):]
        
        return self
    
    def predict(self):
        """
        Делает прогноз на self.horizon шагов вперед
        Используется рекурсивный подход
        """
        predictions = []
        current_series = list(self.last_values)
        current_dates = list(self.last_dates)
        
        for i in range(self.horizon):
            # Создаем дату для следующего шага
            next_date = current_dates[-1] + pd.Timedelta(hours=1)
            
            # Создаем признаки для этого шага
            features = {}
            for lag in self.lags:
                if lag <= len(current_series):
                    features[f'lag_{lag}'] = current_series[-lag]
                else:
                    features[f'lag_{lag}'] = np.nan
            
            features['hour'] = next_date.hour
            features['dayofweek'] = next_date.dayofweek
            
            # Преобразуем в DataFrame
            X_pred = pd.DataFrame([features])
            
            # Предсказываем
            pred = self.model.predict(X_pred)[0]
            predictions.append(pred)
            
            # Обновляем текущий ряд для следующей итерации
            current_series.append(pred)
            current_dates.append(next_date)
        
        return np.array(predictions)


def evaluate_model(
    df: pd.DataFrame,
    windows_by_id: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    model_class,
    model_params: dict,
    horizon: int = 48
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Универсальная функция для оценки модели на всех окнах с прогресc-баром
    """
    predictions_by_window = {i: [] for i in range(len(next(iter(windows_by_id.values()))))}
    targets_by_window = {i: [] for i in range(len(next(iter(windows_by_id.values()))))}
    
    unique_ids = df['unique_id'].unique()
    print(f"Всего рядов для обработки: {len(unique_ids)}")
    
    # Создаем прогресс-бар для рядов
    for unique_id in tqdm(unique_ids, desc="Обработка рядов"):
        series_df = df[df['unique_id'] == unique_id].reset_index(drop=True)
        series = series_df['y'].values
        dates = series_df['ds'].values
        windows = windows_by_id[unique_id]
        
        for window_idx, (train_idx, test_idx) in enumerate(windows):
            train = series[train_idx]
            train_dates = dates[train_idx]
            test = series[test_idx]
            
            # Обучаем модель
            model = model_class(horizon=horizon, **model_params)
            model.fit(train, pd.DatetimeIndex(train_dates))
            
            # Предсказываем
            pred = model.predict()
            
            # Сохраняем
            predictions_by_window[window_idx].extend(pred)
            targets_by_window[window_idx].extend(test)
    
    # Преобразуем в numpy
    for window_idx in predictions_by_window:
        predictions_by_window[window_idx] = np.array(predictions_by_window[window_idx])
        targets_by_window[window_idx] = np.array(targets_by_window[window_idx])
    
    return predictions_by_window, targets_by_window