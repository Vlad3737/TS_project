import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

class SimpleEnsemble:
    """
    Простое усреднение предсказаний нескольких моделей
    """
    def __init__(self, models_predictions: Dict[str, Dict[int, np.ndarray]]):
        """
        models_predictions: {
            'model_name': {
                0: предсказания для окна 0,
                1: предсказания для окна 1,
                ...
            }
        }
        """
        self.models_predictions = models_predictions
        self.model_names = list(models_predictions.keys())
        
    def predict(self, weights: List[float] = None):
        """
        Взвешенное усреднение предсказаний
        Если weights=None, то равные веса
        """
        n_models = len(self.model_names)
        if weights is None:
            weights = [1.0/n_models] * n_models
        
        # Нормализуем веса
        weights = np.array(weights) / sum(weights)
        
        # Получаем все окна
        all_windows = set()
        for model_preds in self.models_predictions.values():
            all_windows.update(model_preds.keys())
        
        ensemble_preds = {}
        for window in all_windows:
            # Собираем предсказания всех моделей для этого окна
            window_preds = []
            for i, model_name in enumerate(self.model_names):
                if window in self.models_predictions[model_name]:
                    preds = self.models_predictions[model_name][window]
                    window_preds.append(preds * weights[i])
            
            # Усредняем
            ensemble_preds[window] = np.sum(window_preds, axis=0)
        
        return ensemble_preds


class StackingEnsemble:
    """
    Стекинг: обучаем мета-модель на предсказаниях базовых моделей
    """
    def __init__(
        self, 
        base_predictions: Dict[str, Dict[int, np.ndarray]],
        targets: Dict[int, np.ndarray],
        window_ids: List[int],
        meta_model=None
    ):
        """
        base_predictions: предсказания базовых моделей по окнам
        targets: истинные значения
        window_ids: список ID окон для обучения мета-модели
        """
        self.base_predictions = base_predictions
        self.targets = targets
        self.window_ids = window_ids
        self.meta_model = meta_model or LinearRegression()
        self.weights_ = None
        
    def fit(self, val_window_ids: List[int]):
        """
        Обучаем мета-модель на валидационных окнах
        """
        # Собираем обучающие данные
        X_train = []
        y_train = []
        
        for window in val_window_ids:
            # Собираем предсказания всех моделей для этого окна
            window_features = []
            for model_name in self.base_predictions.keys():
                if window in self.base_predictions[model_name]:
                    preds = self.base_predictions[model_name][window]
                    window_features.append(preds)
            
            if window_features and window in self.targets:
                X_train.append(np.column_stack(window_features))
                y_train.append(self.targets[window])
        
        X_train = np.vstack(X_train)
        y_train = np.concatenate(y_train)
        
        # Обучаем мета-модель
        self.meta_model.fit(X_train, y_train)
        
        # Сохраняем веса (для линейной регрессии)
        if hasattr(self.meta_model, 'coef_'):
            self.weights_ = self.meta_model.coef_
        
        return self
    
    def predict(self, test_window_ids: List[int]) -> Dict[int, np.ndarray]:
        """
        Предсказываем для тестовых окон
        """
        predictions = {}
        
        for window in test_window_ids:
            # Собираем предсказания всех моделей
            window_features = []
            for model_name in self.base_predictions.keys():
                if window in self.base_predictions[model_name]:
                    preds = self.base_predictions[model_name][window]
                    window_features.append(preds)
            
            if window_features:
                X_test = np.column_stack(window_features)
                predictions[window] = self.meta_model.predict(X_test)
        
        return predictions


def evaluate_ensemble(
    ensemble_preds: Dict[int, np.ndarray],
    targets: Dict[int, np.ndarray],
    window_ids: List[int],
    ensemble_name: str
):
    """
    Оценивает качество ансамбля
    """
    from src.utils import calculate_metrics
    
    results = []
    for window in window_ids:
        if window in ensemble_preds and window in targets:
            metrics = calculate_metrics(
                targets[window], 
                ensemble_preds[window]
            )
            metrics['window'] = window
            metrics['model'] = ensemble_name
            results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    # Считаем статистики
    summary = {
        'model': ensemble_name,
        'mean_smape': df_results['smape'].mean(),
        'std_smape': df_results['smape'].std(),
        'mean_mae': df_results['mae'].mean(),
        'std_mae': df_results['mae'].std(),
        'min_smape': df_results['smape'].min(),
        'max_smape': df_results['smape'].max()
    }
    
    return df_results, pd.DataFrame([summary])