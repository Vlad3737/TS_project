import pandas as pd

# Ссылки на данные M4 Hourly
M4_INFO_URL = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/M4-info.csv"
M4_HOURLY_URL = "https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/Train/Hourly-train.csv"

def download_m4_hourly(save_path="data/m4_hourly.csv"):
    metadata = pd.read_csv(M4_INFO_URL)
    metadata = metadata[metadata["SP"] == "Hourly"].set_index("M4id")

    data = pd.read_csv(M4_HOURLY_URL, index_col="V1")

    results = []
    for item_id in metadata.index:
        # Берем ряд, убираем пропуски (NaN), которые могут быть в конце
        time_series = data.loc[item_id].dropna().values
        start_time = pd.Timestamp(metadata.loc[item_id]["StartingDate"])
        # Создаем временные метки (почасовая частота)
        timestamps = pd.date_range(start_time, freq="H", periods=len(time_series))
        # Создаем DataFrame в формате "id", "timestamp", "value"
        df_item = pd.DataFrame({
            "unique_id": [item_id] * len(time_series),
            "ds": timestamps,
            "y": time_series
        })
        results.append(df_item)

    result_df = pd.concat(results, ignore_index=True)
    result_df.to_csv(save_path, index=False)
    print(f"Данные сохранены в {save_path}. Всего строк: {len(result_df)}")
    return result_df

# Создаем папку data, если её нет
import os
os.makedirs("data", exist_ok=True)

# Скачиваем
df = download_m4_hourly()
print(df.head())