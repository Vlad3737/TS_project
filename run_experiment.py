import subprocess
import sys

# Загрузка данных
print("\n1. Загрузка данных...")
subprocess.run([sys.executable, "-c", 
    "from src.data_loader import download_m4_hourly; download_m4_hourly()"])

# Запуск бейзлайнов и CatBoost
print("\n2. Запуск основных экспериментов...")
subprocess.run([sys.executable, "test_catboost.py"])

# Запуск ансамблей
print("\n3. Запуск ансамблей...")
subprocess.run([sys.executable, "test_ensemble.py"])

print("\n✅ Все эксперименты завершены!")
print("Результаты в папке results/")