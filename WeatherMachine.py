import sys
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class WeatherScraper:
    def __init__(self):
        # 多个数据源配置
        self.data_sources = {
            'open_meteo': {
                'name': 'Open-Meteo',
                'url': 'https://archive-api.open-meteo.com/v1/archive',
                'free': True,
                'description': '免费开源天气API'
            },
            'openweather': {
                'name': 'OpenWeatherMap',
                'url': 'https://history.openweathermap.org/data/2.5/history/city',
                'free': False,  # 需要API key
                'description': '全球知名天气API（需要API key）'
            },
            'weatherapi': {
                'name': 'WeatherAPI',
                'url': 'http://api.weatherapi.com/v1/history.json',
                'free': False,  # 需要API key
                'description': '专业天气API（需要API key）'
            },
            'visual_crossing': {
                'name': 'Visual Crossing',
                'url': 'https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline',
                'free': False,  # 需要API key
                'description': '可视化天气数据API（需要API key）'
            }
        }
        
        # API Keys (用户可以在这里配置自己的API密钥)
        self.api_keys = {
            'openweather': '',  # 在这里填入OpenWeatherMap的API key
            'weatherapi': '',   # 在这里填入WeatherAPI的API key
            'visual_crossing': ''  # 在这里填入Visual Crossing的API key
        }

    def get_city_coordinates(self, city_name):
        city_coords = {
            "北京": (39.90, 116.41),
            "上海": (31.23, 121.47),
            "广州": (23.13, 113.26),
            "深圳": (22.54, 114.05),
            "西安": (34.34, 108.94),
        }
        return city_coords.get(city_name, None)

    def fetch_weather_data(self, city_name, days_back=1825):
        coords = self.get_city_coordinates(city_name)
        if not coords:
            raise ValueError(f"不支持的城市：{city_name}")

        print(f"🌍 正在为 {city_name} 获取天气数据...")
        
        # 按优先级尝试不同数据源
        data_source_priority = ['open_meteo', 'weatherapi', 'visual_crossing', 'openweather']
        
        for source_name in data_source_priority:
            source_info = self.data_sources[source_name]
            print(f"\n📡 尝试数据源: {source_info['name']} ({source_info['description']})")
            
            try:
                if source_name == 'open_meteo':
                    df = self._fetch_from_open_meteo(city_name, coords, days_back)
                elif source_name == 'weatherapi':
                    df = self._fetch_from_weatherapi(city_name, coords, days_back)
                elif source_name == 'visual_crossing':
                    df = self._fetch_from_visual_crossing(city_name, coords, days_back)
                elif source_name == 'openweather':
                    df = self._fetch_from_openweather(city_name, coords, days_back)
                
                if df is not None and len(df) > 100:  # 确保获取到足够数据
                    print(f"✅ 成功从 {source_info['name']} 获取 {len(df)} 天的数据")
                    self._save_cache(df, city_name)
                    return df
                else:
                    print(f"❌ {source_info['name']} 返回数据不足")
                    
            except Exception as e:
                print(f"❌ {source_info['name']} 获取失败: {str(e)}")
                continue
        
        # 所有数据源都失败，尝试本地缓存或生成模拟数据
        print("\n🔄 所有在线数据源都不可用，尝试备用方案...")
        return self._try_fallback_data(city_name, days_back)

    def _fetch_from_open_meteo(self, city_name, coords, days_back):
        """从Open-Meteo获取数据"""
        latitude, longitude = coords
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days_back)

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "temperature_2m,precipitation,surface_pressure,windspeed_10m,relative_humidity_2m,cloudcover",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum"
        }

        url = self.data_sources['open_meteo']['url']
        
        # 重试机制
        for attempt in range(3):
            try:
                print(f"  🔄 尝试连接 Open-Meteo (第 {attempt + 1}/3 次)...")
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    return self._process_open_meteo_data(data)
                else:
                    print(f"  ❌ HTTP错误: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"  ❌ 连接错误: {e}")
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
        
        return None

    def _fetch_from_weatherapi(self, city_name, coords, days_back):
        """从WeatherAPI获取数据"""
        if not self.api_keys.get('weatherapi'):
            print("  ⚠️ WeatherAPI需要API key，跳过")
            return None
            
        latitude, longitude = coords
        # WeatherAPI的历史数据获取方法
        # 注意：这里只是示例，实际使用需要API key
        print("  ⚠️ WeatherAPI需要有效的API key才能使用")
        return None

    def _fetch_from_visual_crossing(self, city_name, coords, days_back):
        """从Visual Crossing获取数据"""
        if not self.api_keys.get('visual_crossing'):
            print("  ⚠️ Visual Crossing需要API key，跳过")
            return None
            
        # Visual Crossing的API调用示例
        print("  ⚠️ Visual Crossing需要有效的API key才能使用")
        return None

    def _fetch_from_openweather(self, city_name, coords, days_back):
        """从OpenWeatherMap获取数据"""
        if not self.api_keys.get('openweather'):
            print("  ⚠️ OpenWeatherMap需要API key，跳过")
            return None
            
        # OpenWeatherMap的历史数据API调用
        print("  ⚠️ OpenWeatherMap需要有效的API key才能使用")
        return None

    def _process_open_meteo_data(self, data):
        """处理Open-Meteo数据格式"""
        # 使用每日数据获取更准确的最高最低温度
        daily_df = pd.DataFrame({
            "date": pd.to_datetime(data["daily"]["time"]),
            "high_temp": data["daily"]["temperature_2m_max"],
            "low_temp": data["daily"]["temperature_2m_min"],
            "rain": data["daily"]["precipitation_sum"]
        })

        # 从小时数据获取其他特征的日平均值
        hourly_df = pd.DataFrame({
            "datetime": pd.to_datetime(data["hourly"]["time"]),
            "pressure": data["hourly"]["surface_pressure"],
            "wind_speed": data["hourly"]["windspeed_10m"],
            "humidity": data["hourly"]["relative_humidity_2m"],
            "cloudcover": data["hourly"]["cloudcover"]
        })

        # 按日期聚合小时数据
        hourly_df['date'] = hourly_df['datetime'].dt.date
        hourly_agg = hourly_df.groupby('date').agg({
            "pressure": "mean",
            "wind_speed": "mean",
            "humidity": "mean",
            "cloudcover": "mean"
        }).reset_index()
        hourly_agg['date'] = pd.to_datetime(hourly_agg['date'])

        # 合并日数据和小时聚合数据
        df = pd.merge(daily_df, hourly_agg, on='date', how='inner')
        
        # 计算平均温度
        df['avg_temp'] = (df['high_temp'] + df['low_temp']) / 2

        return df

    def _try_fallback_data(self, city_name, days_back):
        """尝试使用备用数据源或本地缓存"""
        # 首先尝试读取本地缓存
        cache_file = f"weather_cache_{city_name}.csv"
        if os.path.exists(cache_file):
            print(f"📁 找到本地缓存文件: {cache_file}")
            try:
                df = pd.read_csv(cache_file)
                df['date'] = pd.to_datetime(df['date'])
                print(f"✅ 成功从本地缓存加载 {len(df)} 天的数据")
                return df
            except Exception as e:
                print(f"❌ 读取缓存文件失败: {e}")
        
        # 尝试免费的公共数据源
        print("🌐 尝试免费的公共天气数据源...")
        
        # 尝试MeteoStat (另一个免费的天气数据源)
        try:
            df = self._fetch_from_meteostat(city_name, days_back)
            if df is not None:
                return df
        except Exception as e:
            print(f"❌ MeteoStat获取失败: {e}")
        
        # 如果都失败，生成模拟数据用于测试
        print("🔧 所有数据源都不可用，生成模拟数据用于测试...")
        return self._generate_mock_data(city_name, days_back)

    def _fetch_from_meteostat(self, city_name, days_back):
        """尝试从MeteoStat获取数据（免费的气象数据库）"""
        try:
            # 注意：这需要安装meteostat库: pip install meteostat
            import meteostat
            from meteostat import Point, Daily
            
            coords = self.get_city_coordinates(city_name)
            latitude, longitude = coords
            
            end_date = datetime.datetime.now()
            start_date = end_date - datetime.timedelta(days=days_back)
            
            print("  🔄 尝试连接 MeteoStat...")
            
            # 创建地点对象
            location = Point(latitude, longitude)
            
            # 获取每日数据
            data = Daily(location, start_date, end_date)
            data = data.fetch()
            
            if len(data) > 100:
                df = pd.DataFrame({
                    'date': data.index,
                    'high_temp': data['tmax'],
                    'low_temp': data['tmin'],
                    'avg_temp': data['tavg'],
                    'rain': data['prcp'].fillna(0),
                    'pressure': data['pres'].fillna(1013),
                    'wind_speed': data['wspd'].fillna(0),
                    'humidity': data.get('rhum', pd.Series([50] * len(data))),  # 默认湿度50%
                    'cloudcover': data.get('tsun', pd.Series([50] * len(data)))  # 用日照时间估算云量
                })
                
                # 填充缺失的avg_temp
                df['avg_temp'] = df['avg_temp'].fillna((df['high_temp'] + df['low_temp']) / 2)
                
                # 重置索引
                df = df.reset_index(drop=True)
                
                print(f"✅ MeteoStat 成功获取 {len(df)} 天的数据")
                return df
            else:
                print("❌ MeteoStat 数据不足")
                return None
                
        except ImportError:
            print("  ⚠️ MeteoStat库未安装，运行: pip install meteostat")
            return None
        except Exception as e:
            print(f"  ❌ MeteoStat错误: {e}")
            return None

    def _save_cache(self, df, city_name):
        """保存数据到本地缓存"""
        cache_file = f"weather_cache_{city_name}.csv"
        try:
            df.to_csv(cache_file, index=False)
            print(f"💾 数据已缓存到: {cache_file}")
        except Exception as e:
            print(f"⚠️ 缓存保存失败: {e}")

    def _generate_mock_data(self, city_name, days_back):
        """生成模拟天气数据用于测试"""
        print("🎭 正在生成模拟天气数据...")
        
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=days_back)
        
        # 生成日期序列
        dates = pd.date_range(start=start_date, end=end_date, freq='D')[:-1]  # 去掉最后一天
        
        # 根据城市设置基础温度
        city_base_temps = {
            "北京": 15, "上海": 18, "广州": 22, "深圳": 24, "西安": 16
        }
        base_temp = city_base_temps.get(city_name, 18)
        
        # 生成模拟数据
        np.random.seed(42)  # 固定随机种子
        n_days = len(dates)
        
        # 温度数据（考虑季节性）
        seasonal_temp = 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 - np.pi/2)
        daily_variation = np.random.normal(0, 3, n_days)
        avg_temps = base_temp + seasonal_temp + daily_variation
        
        high_temps = avg_temps + np.random.uniform(3, 8, n_days)
        low_temps = avg_temps - np.random.uniform(3, 8, n_days)
        
        # 降雨数据（偏向小雨和无雨）
        rain_prob = np.random.random(n_days)
        rain_amounts = np.where(rain_prob < 0.7, 0, np.random.exponential(2, n_days))  # 70%无雨
        
        # 其他气象数据
        pressures = np.random.normal(1013, 20, n_days)
        wind_speeds = np.random.exponential(3, n_days)
        humidities = np.random.uniform(30, 90, n_days)
        cloudcovers = np.random.uniform(0, 100, n_days)
        
        df = pd.DataFrame({
            'date': dates,
            'high_temp': high_temps,
            'low_temp': low_temps,
            'avg_temp': (high_temps + low_temps) / 2,
            'rain': rain_amounts,
            'pressure': pressures,
            'wind_speed': wind_speeds,
            'humidity': humidities,
            'cloudcover': cloudcovers
        })
        
        print(f"✅ 生成了 {len(df)} 天的模拟数据")
        print("⚠️ 注意: 这是模拟数据，仅用于测试系统功能")
        
        return df

    def list_available_sources(self):
        """列出所有可用的数据源"""
        print("\n🌐 可用的天气数据源:")
        print("=" * 60)
        
        for source_name, info in self.data_sources.items():
            status = "✅ 免费" if info['free'] else "🔑 需要API key"
            has_key = "✅ 已配置" if self.api_keys.get(source_name) else "❌ 未配置"
            
            print(f"{info['name']:20} | {status:10} | {has_key:10} | {info['description']}")
        
        print("\n💡 提示:")
        print("1. Open-Meteo 是完全免费的，无需API key")
        print("2. 其他服务需要注册并获取API key")
        print("3. 大多数服务都有免费额度，足够个人使用")
        print("4. 配置多个数据源可以提高系统可靠性")

    def set_api_key(self, source_name, api_key):
        """设置API密钥"""
        if source_name in self.api_keys:
            self.api_keys[source_name] = api_key
            print(f"✅ 已设置 {source_name} 的API key")
        else:
            print(f"❌ 不支持的数据源: {source_name}")

class WeatherDataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_data(self, df, city="未知城市", enable_visualization=False):
        processed_df = df.copy()
        processed_df['date'] = pd.to_datetime(processed_df['date'])
        processed_df['year'] = processed_df['date'].dt.year
        processed_df['month'] = processed_df['date'].dt.month
        processed_df['day'] = processed_df['date'].dt.day
        processed_df['day_of_week'] = processed_df['date'].dt.dayofweek
        
        # 添加更多特征
        processed_df['season'] = processed_df['month'].apply(self._get_season)
        processed_df['weekend_flag'] = (processed_df['day_of_week'] >= 5).astype(int)
        
        # 添加移动平均特征
        processed_df['temp_ma_3'] = processed_df['avg_temp'].rolling(window=3, min_periods=1).mean()
        processed_df['temp_ma_7'] = processed_df['avg_temp'].rolling(window=7, min_periods=1).mean()
        processed_df['rain_ma_3'] = processed_df['rain'].rolling(window=3, min_periods=1).mean()
        
        # 添加趋势特征
        processed_df['temp_trend'] = processed_df['avg_temp'].diff().fillna(0)
        processed_df['pressure_change'] = processed_df['pressure'].diff().fillna(0)
        processed_df['humidity_change'] = processed_df['humidity'].diff().fillna(0)
        
        processed_df.dropna(inplace=True)
        return processed_df
    
    def _get_season(self, month):
        if month in [12, 1, 2]:
            return 0  # 冬季
        elif month in [3, 4, 5]:
            return 1  # 春季
        elif month in [6, 7, 8]:
            return 2  # 夏季
        else:
            return 3  # 秋季

    def create_sequences(self, data, feature_cols, target_cols, seq_length=7, forecast_days=3):
        """支持多目标输出的序列创建"""
        X, y = [], []
        for i in range(len(data) - seq_length - forecast_days):
            X.append(data[feature_cols].iloc[i:i + seq_length].values)
            if isinstance(target_cols, list):
                # 多目标输出
                y.append(data[target_cols].iloc[i + seq_length:i + seq_length + forecast_days].values)
            else:
                # 单目标输出
                y.append(data[target_cols].iloc[i + seq_length:i + seq_length + forecast_days].values)
        return np.array(X), np.array(y)

    def split_train_val_test_by_time(self, X, y, df, seq_length=7, forecast_days=3):
        val_start_date = df['date'].max() - pd.DateOffset(years=2)
        test_start_date = df['date'].max() - pd.DateOffset(years=1)

        val_indices = df[(df['date'] >= val_start_date) & (df['date'] < test_start_date)].index
        test_indices = df[df['date'] >= test_start_date].index

        train_end_idx = val_indices.min() - seq_length - forecast_days
        val_end_idx = test_indices.min() - seq_length - forecast_days

        X_train = X[:train_end_idx]
        y_train = y[:train_end_idx]
        X_val = X[train_end_idx:val_end_idx]
        y_val = y[train_end_idx:val_end_idx]
        X_test = X[val_end_idx:]
        y_test = y[val_end_idx:]

        print("\n=== 数据划分详情 ===")
        print(f"总样本数: {len(X)}")
        print(f"训练集样本数: {len(X_train)}")
        print(f"验证集样本数: {len(X_val)}")
        print(f"测试集样本数: {len(X_test)}")
        print(f"验证集起始日期: {val_start_date.date()}")
        if len(X_val) > 0:
            print(f"验证集覆盖时间范围: {df.iloc[train_end_idx]['date']} 到 {df.iloc[val_end_idx - 1]['date']}")
        if len(X_test) > 0:
            print(f"测试集起始日期: {test_start_date.date()}")
            print(f"测试集覆盖时间范围: {df.iloc[val_end_idx]['date']} 到 {df.iloc[-forecast_days - 1]['date']}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def scale_data(self, X_train, X_val, X_test):
        n_samples_train, n_steps, n_features = X_train.shape
        n_samples_val = X_val.shape[0] if len(X_val) > 0 else 0
        n_samples_test = X_test.shape[0] if len(X_test) > 0 else 0

        X_train_flat = X_train.reshape(-1, n_features)
        X_train_scaled = self.scaler.fit_transform(X_train_flat).reshape(n_samples_train, n_steps, n_features)
        
        if n_samples_val > 0:
            X_val_flat = X_val.reshape(-1, n_features)
            X_val_scaled = self.scaler.transform(X_val_flat).reshape(n_samples_val, n_steps, n_features)
        else:
            X_val_scaled = np.array([])
            
        if n_samples_test > 0:
            X_test_flat = X_test.reshape(-1, n_features)
            X_test_scaled = self.scaler.transform(X_test_flat).reshape(n_samples_test, n_steps, n_features)
        else:
            X_test_scaled = np.array([])

        return X_train_scaled, X_val_scaled, X_test_scaled

# 多输出温度预测模型
class MultiOutputTemperaturePredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, forecast_days, num_layers=3, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads=8, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_dim*2)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # 三个输出分支：最高温、最低温、平均温
        self.fc_high = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, forecast_days)
        )
        
        self.fc_low = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, forecast_days)
        )
        
        self.fc_avg = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, forecast_days)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        
        # 注意力机制
        lstm_out_t = lstm_out.transpose(0, 1)
        attn_out, _ = self.attention(lstm_out_t, lstm_out_t, lstm_out_t)
        attn_out = attn_out.transpose(0, 1)
        attn_out = self.norm1(attn_out + lstm_out)
        
        # 使用最后一个时间步的输出
        last_hidden = attn_out[:, -1, :]
        
        # 三个分支输出
        high_temp = self.fc_high(last_hidden)
        low_temp = self.fc_low(last_hidden)
        avg_temp = self.fc_avg(last_hidden)
        
        return {
            'high_temp': high_temp,
            'low_temp': low_temp,
            'avg_temp': avg_temp
        }

# 简化的降雨预测模型
class SimpleRainfallPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, forecast_days, num_layers=2, dropout=0.3):
        super().__init__()
        
        # CNN特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=dropout),
            num_layers=num_layers
        )
        
        # 直接预测降雨量（回归任务）
        self.rain_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, forecast_days),
            nn.ReLU()  # 确保输出非负
        )

    def forward(self, x):
        # 特征提取
        x_t = x.transpose(1, 2)
        features = self.feature_extractor(x_t)
        features = features.transpose(1, 2)
        
        # Transformer编码
        encoded_features = self.transformer(features)
        last_hidden = encoded_features[:, -1, :]
        
        # 直接预测降雨量
        rain_amount = self.rain_regressor(last_hidden)
        
        return rain_amount

# 保持向后兼容的两阶段模型（重命名）
class TwoStageRainfallPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, forecast_days, num_layers=2, dropout=0.3):
        super().__init__()
        
        # 共享的特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=dropout),
            num_layers=num_layers
        )
        
        # 第一阶段：预测是否会下雨（分类任务）
        self.rain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_days),
            nn.Sigmoid()  # 输出0-1之间的概率
        )
        
        # 第二阶段：预测降雨量（回归任务）
        self.rain_regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, forecast_days),
            nn.ReLU()  # 确保输出非负
        )

    def forward(self, x):
        # 特征提取
        x_t = x.transpose(1, 2)
        features = self.feature_extractor(x_t)
        features = features.transpose(1, 2)
        
        # Transformer编码
        encoded_features = self.transformer(features)
        last_hidden = encoded_features[:, -1, :]
        
        # 两阶段预测
        rain_prob = self.rain_classifier(last_hidden)  # 降雨概率
        rain_amount = self.rain_regressor(last_hidden)  # 降雨量
        
        return {
            'rain_prob': rain_prob,
            'rain_amount': rain_amount
        }

# 保持原有的模型以向后兼容
class EnhancedLSTMPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=3, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return out

class RainfallPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.ReLU()
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dropout=dropout),
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x = self.transformer(x)
        out = self.fc(x[:, -1, :])
        return out

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def create_model_directory(city):
    """为指定城市创建模型存储目录"""
    model_dir = f"models/{city}"
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def train_and_save(city, forecast_days_list=[3, 5]):
    scraper = WeatherScraper()
    processor = WeatherDataProcessor()
    df = scraper.fetch_weather_data(city)
    df = processor.preprocess_data(df, city=city)
    
    # 更新特征列表，包含新特征
    feature_cols = [
        'year', 'month', 'day', 'day_of_week', 'pressure', 'wind_speed', 
        'humidity', 'cloudcover', 'season', 'weekend_flag',
        'temp_ma_3', 'temp_ma_7', 'rain_ma_3',
        'temp_trend', 'pressure_change', 'humidity_change'
    ]
    
    # 创建城市特定的模型目录
    model_dir = create_model_directory(city)
    
    seq_length = 7

    for forecast_days in forecast_days_list:
        # 训练多输出温度预测模型
        print(f"\n=== 正在为{city}训练多输出温度模型, 预测{forecast_days}天 ===")
        temp_targets = ['high_temp', 'low_temp', 'avg_temp']
        X_temp, y_temp = processor.create_sequences(df, feature_cols, temp_targets, seq_length, forecast_days)
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_train_val_test_by_time(
            X_temp, y_temp, df, seq_length, forecast_days
        )
        X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_data(X_train, X_val, X_test)
        
        # 训练温度模型
        input_dim = X_train_scaled.shape[2]
        temp_model = MultiOutputTemperaturePredictor(
            input_dim=input_dim,
            hidden_dim=64,
            forecast_days=forecast_days,
            num_layers=2,
            dropout=0.3
        )
        
        # 训练过程 - 温度模型
        temp_model = train_multi_output_model(
            temp_model, X_train_scaled, y_train, X_val_scaled, y_val, 
            model_path=f"{model_dir}/temp_model_{forecast_days}days.pth",
            model_type="temperature"
        )
        
        # 训练简化的降雨预测模型
        print(f"\n=== 正在为{city}训练简化降雨模型, 预测{forecast_days}天 ===")
        # 为降雨模型准备数据
        rain_target = 'rain'
        X_rain, y_rain = processor.create_sequences(df, feature_cols, rain_target, seq_length, forecast_days)
        X_train_r, X_val_r, X_test_r, y_train_r, y_val_r, y_test_r = processor.split_train_val_test_by_time(
            X_rain, y_rain, df, seq_length, forecast_days
        )
        X_train_r_scaled, X_val_r_scaled, X_test_r_scaled = processor.scale_data(X_train_r, X_val_r, X_test_r)
        
        rain_model = SimpleRainfallPredictor(
            input_dim=input_dim,
            hidden_dim=64,
            forecast_days=forecast_days,
            num_layers=2,
            dropout=0.3
        )
        
        # 训练过程 - 降雨模型
        rain_model = train_simple_rain_model(
            rain_model, X_train_r_scaled, y_train_r, X_val_r_scaled, y_val_r,
            model_path=f"{model_dir}/rain_model_{forecast_days}days.pth"
        )
        
        print(f"城市 {city} 的模型训练完成并已保存到 {model_dir}")

def train_multi_output_model(model, X_train, y_train, X_val, y_val, model_path, model_type="temperature"):
    """训练多输出模型"""
    if len(X_val) == 0:
        print("警告：验证集为空，跳过验证")
        X_val_tensor = None
        y_val_tensor = None
    else:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    train_dataset = DataLoader(
        list(zip(X_train_tensor, y_train_tensor)), 
        batch_size=32, shuffle=True
    )
    
    if X_val_tensor is not None:
        val_dataset = DataLoader(
            list(zip(X_val_tensor, y_val_tensor)), 
            batch_size=32
        )
    else:
        val_dataset = None
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=15)
    
    best_loss = float('inf')
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_dataset:
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # 计算多输出损失
            if model_type == "temperature":
                loss_high = criterion(outputs['high_temp'], y_batch[:, :, 0])  # 最高温
                loss_low = criterion(outputs['low_temp'], y_batch[:, :, 1])    # 最低温
                loss_avg = criterion(outputs['avg_temp'], y_batch[:, :, 2])    # 平均温
                loss = (loss_high + loss_low + loss_avg) / 3
            else:
                loss = criterion(outputs, y_batch)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        if val_dataset is not None:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for X_val_batch, y_val_batch in val_dataset:
                    pred = model(X_val_batch)
                    if model_type == "temperature":
                        val_loss_high = criterion(pred['high_temp'], y_val_batch[:, :, 0])
                        val_loss_low = criterion(pred['low_temp'], y_val_batch[:, :, 1])
                        val_loss_avg = criterion(pred['avg_temp'], y_val_batch[:, :, 2])
                        val_loss = (val_loss_high + val_loss_low + val_loss_avg) / 3
                    else:
                        val_loss = criterion(pred, y_val_batch)
                    val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
            
            print(f'Epoch {epoch + 1}, Train Loss: {total_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), model_path)
            
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("早停触发，结束训练")
                break
        else:
            print(f'Epoch {epoch + 1}, Train Loss: {total_loss:.4f}')
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), model_path)
    
    print(f"模型训练完成并已保存到 {model_path}")
    return model

def train_simple_rain_model(model, X_train, y_train, X_val, y_val, model_path, rain_threshold=5.0):
    """训练简化的降雨模型 - 直接预测降雨量"""
    if len(X_val) == 0:
        print("警告：验证集为空，跳过验证")
        X_val_tensor = None
        y_val_tensor = None
    else:
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    
    train_dataset = DataLoader(
        list(zip(X_train_tensor, y_train_tensor)), 
        batch_size=32, shuffle=True
    )
    
    if X_val_tensor is not None:
        val_dataset = DataLoader(
            list(zip(X_val_tensor, y_val_tensor)), 
            batch_size=32
        )
    else:
        val_dataset = None
    
    # 使用MSE损失函数进行回归
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    early_stopping = EarlyStopping(patience=15)
    
    best_loss = float('inf')
    
    for epoch in range(100):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_dataset:
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # 直接使用MSE损失
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # 验证
        if val_dataset is not None:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for X_val_batch, y_val_batch in val_dataset:
                    pred = model(X_val_batch)
                    val_loss = criterion(pred, y_val_batch)
                    val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
            
            print(f'Epoch {epoch + 1}, Train Loss: {total_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(model.state_dict(), model_path)
            
            scheduler.step(avg_val_loss)
            early_stopping(avg_val_loss)
            if early_stopping.early_stop:
                print("早停触发，结束训练")
                break
        else:
            print(f'Epoch {epoch + 1}, Train Loss: {total_loss:.4f}')
            if total_loss < best_loss:
                best_loss = total_loss
                torch.save(model.state_dict(), model_path)
    
    print(f"简化降雨模型训练完成并已保存到 {model_path}")
    return model

if __name__ == "__main__":
    city = input("请输入城市名称（如 西安、北京、上海）：")
    train_and_save(city, forecast_days_list=[3, 5])  # 可根据需要添加更多天数