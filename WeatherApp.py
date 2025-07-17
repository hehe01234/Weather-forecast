import sys
sys.path.append(r"D:\大学\人工智能\达内实习\项目\Weather")

import streamlit as st
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

from WeatherMachine import (
    WeatherScraper, WeatherDataProcessor,
    MultiOutputTemperaturePredictor, SimpleRainfallPredictor,
    EnhancedLSTMPredictor, RainfallPredictor  # 保持向后兼容
)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="天气预测AI系统", layout="wide")
st.title("🌦️ 智能天气预测系统 - 增强版")
st.markdown("""
本系统基于深度学习模型，支持：
- 🌡️ **多维度温度预测**：同时预测最高温度、最低温度和平均温度
- 🌧️ **智能降雨预测**：先判断是否下雨，再预测降雨量
- 🏙️ **城市专属模型**：为每个城市训练专用模型
- 📊 **丰富特征工程**：云量、季节、移动平均等15+特征
""")

# 侧边栏配置
st.sidebar.header("预测配置")
city = st.sidebar.selectbox("选择城市", ["北京", "上海", "广州", "深圳", "西安"])
forecast_days = st.sidebar.selectbox("预测未来天数", [3, 5])
target = st.sidebar.selectbox("预测目标", ["温度", "降雨量"])

def check_model_exists(city, model_type, forecast_days):
    """检查指定城市的模型是否存在"""
    model_path = f"models/{city}/{model_type}_model_{forecast_days}days.pth"
    return os.path.exists(model_path)

def get_feature_columns():
    """获取增强的特征列表"""
    return [
        'year', 'month', 'day', 'day_of_week', 'pressure', 'wind_speed', 
        'humidity', 'cloudcover', 'season', 'weekend_flag',
        'temp_ma_3', 'temp_ma_7', 'rain_ma_3',
        'temp_trend', 'pressure_change', 'humidity_change'
    ]

@st.cache_resource(show_spinner=True)
def load_data_and_model(city, forecast_days, target):
    """加载数据和城市专属模型"""
    try:
        scraper = WeatherScraper()
        processor = WeatherDataProcessor()
        df = scraper.fetch_weather_data(city)
        df = processor.preprocess_data(df, city=city)
        feature_cols = get_feature_columns()

        # 检查模型是否存在
        if target == "温度":
            model_type = "temp"
            model_exists = check_model_exists(city, model_type, forecast_days)
        else:
            model_type = "rain"
            model_exists = check_model_exists(city, model_type, forecast_days)
        
        if not model_exists:
            st.error(f"❌ {city}的{target}预测模型（{forecast_days}天）不存在！")
            st.info("💡 请先运行训练脚本生成模型：`python WeatherMachine.py`")
            return None, None, None, None, None, None, None, None

        if target == "温度":
            # 温度预测：多目标输出
            target_cols = ['high_temp', 'low_temp', 'avg_temp']
            X, y = processor.create_sequences(df, feature_cols, target_cols, 7, forecast_days)
            model_path = f"models/{city}/temp_model_{forecast_days}days.pth"
            
            input_dim = len(feature_cols)
            model = MultiOutputTemperaturePredictor(
                input_dim=input_dim,
                hidden_dim=64,
                forecast_days=forecast_days,
                num_layers=2,
                dropout=0.3
            )
        else:
            # 降雨预测：简化模型
            target_col = 'rain'
            X, y = processor.create_sequences(df, feature_cols, target_col, 7, forecast_days)
            model_path = f"models/{city}/rain_model_{forecast_days}days.pth"
            
            input_dim = len(feature_cols)
            model = SimpleRainfallPredictor(
                input_dim=input_dim,
                hidden_dim=64,
                forecast_days=forecast_days,
                num_layers=2,
                dropout=0.3
            )

        # 加载模型权重
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()

        # 数据分割和标准化
        X_train, X_val, X_test, y_train, y_val, y_test = processor.split_train_val_test_by_time(
            X, y, df, 7, forecast_days
        )
        X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_data(X_train, X_val, X_test)

        return df, processor, feature_cols, model, X_val_scaled, y_val, X_test_scaled, y_test

    except Exception as e:
        st.error(f"❌ 加载数据或模型时出错：{str(e)}")
        return None, None, None, None, None, None, None, None

# 加载数据和模型
data_result = load_data_and_model(city, forecast_days, target)
if data_result[0] is None:
    st.stop()

df, processor, feature_cols, model, X_val_scaled, y_val, X_test_scaled, y_test = data_result

# 显示城市和模型信息
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("当前城市", city)
with col2:
    st.metric("预测天数", f"{forecast_days}天")
with col3:
    st.metric("预测目标", target)

st.header(f"🔮 {city}未来{forecast_days}天{target}预测")

# 显示输入特征
with st.expander("📊 显示输入特征（最近7天数据）", expanded=False):
    recent_data = df.tail(7)[['date'] + feature_cols].copy()
    recent_data['date'] = recent_data['date'].dt.strftime('%Y-%m-%d')
    st.dataframe(recent_data, use_container_width=True)

# 进行预测
recent_seq = df.tail(7)[feature_cols].values
recent_seq_scaled = processor.scaler.transform(recent_seq)
recent_seq_scaled = torch.tensor(recent_seq_scaled, dtype=torch.float32).unsqueeze(0)

with torch.no_grad():
    prediction_result = model(recent_seq_scaled)

# 生成未来日期
future_dates = [df['date'].max() + timedelta(days=i+1) for i in range(forecast_days)]

# 根据预测目标显示结果
if target == "温度":
    # 温度预测结果
    high_temps = prediction_result['high_temp'].numpy().flatten()
    low_temps = prediction_result['low_temp'].numpy().flatten()
    avg_temps = prediction_result['avg_temp'].numpy().flatten()
    
    # 创建预测结果表格
    future_df = pd.DataFrame({
        "日期": [d.strftime("%Y-%m-%d") for d in future_dates],
        "最高温度 (°C)": [f"{temp:.1f}" for temp in high_temps],
        "最低温度 (°C)": [f"{temp:.1f}" for temp in low_temps],
        "平均温度 (°C)": [f"{temp:.1f}" for temp in avg_temps]
    })
    
    st.subheader("🌡️ 温度预测结果")
    st.dataframe(future_df, use_container_width=True)
    
    # 温度预测可视化
    fig, ax = plt.subplots(figsize=(12, 6))
    x_pos = range(len(future_dates))
    
    ax.fill_between(x_pos, low_temps, high_temps, alpha=0.3, color='skyblue', label='温度区间')
    ax.plot(x_pos, high_temps, 'r-o', label='最高温度', linewidth=2, markersize=6)
    ax.plot(x_pos, avg_temps, 'g-s', label='平均温度', linewidth=2, markersize=6)
    ax.plot(x_pos, low_temps, 'b-^', label='最低温度', linewidth=2, markersize=6)
    
    ax.set_xlabel('日期')
    ax.set_ylabel('温度 (°C)')
    ax.set_title(f'{city}未来{forecast_days}天温度预测')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([d.strftime("%m-%d") for d in future_dates], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

else:
    # 降雨预测结果
    rain_amounts_raw = prediction_result.numpy().flatten()
    
    # 设置阈值，小于阈值的降雨量设为0
    rain_threshold = 5.0  # 5mm阈值
    rain_amounts = np.where(rain_amounts_raw < rain_threshold, 0.0, rain_amounts_raw)
    
    # 根据降雨量判断天气状态
    rain_status = []
    for amount in rain_amounts:
        if amount >= 38.0:
            rain_status.append("暴雨")
        elif amount >= 18.0:
            rain_status.append("大雨")
        elif amount >= 10.0:
            rain_status.append("中雨")
        elif amount >= 5.0:
            rain_status.append("小雨")
        else:
            rain_status.append("无降雨")
    
    # 创建预测结果表格
    future_df = pd.DataFrame({
        "日期": [d.strftime("%Y-%m-%d") for d in future_dates],
        "预测降雨量 (mm)": [f"{rain:.1f}" for rain in rain_amounts],
        "天气状态": rain_status,
        "原始预测值": [f"{rain:.2f}" for rain in rain_amounts_raw]  # 显示阈值处理前的值
    })
    
    st.subheader("🌧️ 降雨预测结果")
    st.dataframe(future_df, use_container_width=True)
    
    # 降雨预测可视化
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 原始预测值 vs 阈值处理后的值
    x_pos = range(len(future_dates))
    
    # 第一个子图：对比原始预测和阈值处理后的结果
    bars1 = ax1.bar([x - 0.2 for x in x_pos], rain_amounts_raw, width=0.4, 
                   color='lightblue', alpha=0.7, label='原始预测值')
    bars2 = ax1.bar([x + 0.2 for x in x_pos], rain_amounts, width=0.4, 
                   color='darkblue', alpha=0.8, label='阈值处理后')
    ax1.axhline(y=rain_threshold, color='red', linestyle='--', label=f'阈值线 ({rain_threshold}mm)')
    ax1.set_ylabel('降雨量 (mm)')
    ax1.set_title(f'{city}未来{forecast_days}天降雨量预测（阈值: {rain_threshold}mm）')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 第二个子图：最终降雨量预测
    colors = ['darkblue' if r > 0 else 'lightgray' for r in rain_amounts]
    bars3 = ax2.bar(x_pos, rain_amounts, color=colors)
    ax2.set_xlabel('日期')
    ax2.set_ylabel('降雨量 (mm)')
    ax2.set_title(f'{city}未来{forecast_days}天最终降雨量预测')
    
    # 添加降雨强度分级线
    ax2.axhline(y=5.0, color='green', linestyle=':', alpha=0.7, label='小雨阈值(5mm)')
    ax2.axhline(y=10.0, color='orange', linestyle=':', alpha=0.7, label='中雨阈值(10mm)')
    ax2.axhline(y=18.0, color='red', linestyle=':', alpha=0.7, label='大雨阈值(18mm)')
    ax2.axhline(y=38.0, color='purple', linestyle=':', alpha=0.7, label='暴雨阈值(38mm)')
    ax2.legend()
    
    for ax in [ax1, ax2]:
        ax.set_xticks(x_pos)
        ax.set_xticklabels([d.strftime("%m-%d") for d in future_dates], rotation=45)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)

# 模型性能评估
st.header("📈 模型性能评估")

def plot_performance(y_val, X_val_scaled, y_test, X_test_scaled, model, target_type):
    """绘制模型性能"""
    if len(X_val_scaled) == 0 or len(X_test_scaled) == 0:
        st.warning("⚠️ 验证集或测试集数据不足，无法显示性能评估")
        return
    
    with torch.no_grad():
        val_pred = model(torch.tensor(X_val_scaled, dtype=torch.float32))
        test_pred = model(torch.tensor(X_test_scaled, dtype=torch.float32))
    
    if target_type == "温度":
        # 多输出温度模型的性能评估
        metrics_data = []
        temp_types = ['最高温度', '最低温度', '平均温度']
        pred_keys = ['high_temp', 'low_temp', 'avg_temp']
        
        for i, (temp_type, pred_key) in enumerate(zip(temp_types, pred_keys)):
            val_true = y_val[:, :, i].flatten()
            test_true = y_test[:, :, i].flatten()
            val_pred_i = val_pred[pred_key].numpy().flatten()
            test_pred_i = test_pred[pred_key].numpy().flatten()
            
            # 计算指标
            val_mse = mean_squared_error(val_true, val_pred_i)
            val_mae = mean_absolute_error(val_true, val_pred_i)
            val_r2 = r2_score(val_true, val_pred_i)
            
            test_mse = mean_squared_error(test_true, test_pred_i)
            test_mae = mean_absolute_error(test_true, test_pred_i)
            test_r2 = r2_score(test_true, test_pred_i)
            
            metrics_data.append({
                '温度类型': temp_type,
                '验证集MSE': f'{val_mse:.4f}',
                '验证集MAE': f'{val_mae:.4f}',
                '验证集R²': f'{val_r2:.4f}',
                '测试集MSE': f'{test_mse:.4f}',
                '测试集MAE': f'{test_mae:.4f}',
                '测试集R²': f'{test_r2:.4f}'
            })
        
        st.subheader("🎯 温度预测性能指标")
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)
        
        # 可视化一个温度类型的预测效果（平均温度）
        val_true = y_val[:, :, 2].flatten()  # 平均温度
        test_true = y_test[:, :, 2].flatten()
        val_pred_avg = val_pred['avg_temp'].numpy().flatten()
        test_pred_avg = test_pred['avg_temp'].numpy().flatten()
        
    else:
        # 降雨模型的性能评估
        val_true = y_val.flatten()
        test_true = y_test.flatten()
        
        # 获取原始降雨量预测
        val_rain_pred_raw = val_pred.numpy().flatten()
        test_rain_pred_raw = test_pred.numpy().flatten()
        
        # 应用阈值处理
        rain_threshold = 5.0
        val_rain_pred = np.where(val_rain_pred_raw < rain_threshold, 0.0, val_rain_pred_raw)
        test_rain_pred = np.where(test_rain_pred_raw < rain_threshold, 0.0, test_rain_pred_raw)
        
        # 计算分类准确率（是否下雨）
        val_true_class = (val_true > rain_threshold).astype(int)
        test_true_class = (test_true > rain_threshold).astype(int)
        val_pred_class = (val_rain_pred > rain_threshold).astype(int)
        test_pred_class = (test_rain_pred > rain_threshold).astype(int)
        
        val_class_acc = np.mean(val_true_class == val_pred_class)
        test_class_acc = np.mean(test_true_class == test_pred_class)
        
        st.subheader("🎯 降雨预测性能指标")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**验证集性能：**")
            st.write(f"降雨分类准确率: {val_class_acc:.4f}")
            st.write(f"原始预测MSE: {mean_squared_error(val_true, val_rain_pred_raw):.4f}")
            st.write(f"阈值处理后MSE: {mean_squared_error(val_true, val_rain_pred):.4f}")
            st.write(f"降雨量MAE: {mean_absolute_error(val_true, val_rain_pred):.4f}")
            st.write(f"降雨量R²: {r2_score(val_true, val_rain_pred):.4f}")
        
        with col2:
            st.markdown("**测试集性能：**")
            st.write(f"降雨分类准确率: {test_class_acc:.4f}")
            st.write(f"原始预测MSE: {mean_squared_error(test_true, test_rain_pred_raw):.4f}")
            st.write(f"阈值处理后MSE: {mean_squared_error(test_true, test_rain_pred):.4f}")
            st.write(f"降雨量MAE: {mean_absolute_error(test_true, test_rain_pred):.4f}")
            st.write(f"降雨量R²: {r2_score(test_true, test_rain_pred):.4f}")
        
        val_pred_display = val_rain_pred
        test_pred_display = test_rain_pred
    
    # 绘制预测vs真实值对比图
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if target_type == "温度":
        x_val = np.arange(len(val_true))
        x_test = np.arange(len(val_true), len(val_true) + len(test_true))
        
        ax.plot(x_val, val_true, label='验证集真实值', color='blue', alpha=0.7)
        ax.plot(x_val, val_pred_avg, '--', label='验证集预测值', color='orange', alpha=0.7)
        ax.plot(x_test, test_true, label='测试集真实值', color='red', alpha=0.7)
        ax.plot(x_test, test_pred_avg, '--', label='测试集预测值', color='green', alpha=0.7)
        ax.set_ylabel('平均温度 (°C)')
        ax.set_title(f'{city} - 平均温度预测表现（验证集+测试集）')
    else:
        x_val = np.arange(len(val_true))
        x_test = np.arange(len(val_true), len(val_true) + len(test_true))
        
        ax.plot(x_val, val_true, label='验证集真实值', color='blue', alpha=0.7)
        ax.plot(x_val, val_pred_display, '--', label='验证集预测值', color='orange', alpha=0.7)
        ax.plot(x_test, test_true, label='测试集真实值', color='red', alpha=0.7)
        ax.plot(x_test, test_pred_display, '--', label='测试集预测值', color='green', alpha=0.7)
        ax.set_ylabel('降雨量 (mm)')
        ax.set_title(f'{city} - 降雨量预测表现（验证集+测试集）')
    
    ax.set_xlabel('时间步')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

# 调用性能评估函数
plot_performance(y_val, X_val_scaled, y_test, X_test_scaled, model, target)

# 系统信息和提示
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 系统特性")
st.sidebar.markdown("""
- ✅ 城市专属模型
- ✅ 多维度温度预测
- ✅ 智能阈值降雨预测
- ✅ 15+丰富特征
- ✅ 注意力机制
- ✅ 早停与正则化
""")

st.sidebar.markdown("### 📝 使用说明")
st.sidebar.markdown("""
1. 选择城市和预测参数
2. 查看预测结果和可视化
3. 分析模型性能指标
4. 如需重新训练模型，请运行：
   `python WeatherMachine.py`
""")

st.info("💡 **提示**: 如需重新训练模型或添加新城市，请运行后端训练脚本。该系统会自动为每个城市创建专属的温度和降雨预测模型。")

st.markdown("---")
st.markdown("© 2024 智能天气预测系统 | Powered by Streamlit + PyTorch | 增强版 v2.0")