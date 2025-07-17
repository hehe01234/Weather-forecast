# 🌦️ 天气预测AI系统 - 模块化版本 v2.0

## 📋 项目概述

这是一个基于深度学习的智能天气预测系统，采用模块化架构设计，支持多城市、多输出的天气预测。系统可以同时预测温度（最高/最低/平均）和降雨量，并提供分类和回归两种降雨预测模式。

### 🌟 主要特性

- **🏗️ 模块化架构**：清晰的代码结构，易于维护和扩展
- **🏙️ 城市专属模型**：为每个城市单独训练专用模型
- **🌡️ 多维温度预测**：同时预测最高、最低、平均温度
- **🌧️ 智能降雨预测**：支持分类和回归两种预测模式
- **📊 丰富特征工程**：15+特征，包括云量、季节、趋势等
- **🧠 先进模型架构**：LSTM、CNN、Transformer融合
- **📈 完善评估体系**：多种评估指标和可视化
- **🔧 配置管理**：集中化配置管理，支持多环境
- **🧪 单元测试**：全面的测试覆盖
- **📱 Web界面**：Streamlit网页应用

## 📁 项目结构

```
Weather（修改版）/
├── src/                          # 源代码目录
│   ├── __init__.py
│   ├── data_loader/              # 数据加载模块
│   │   ├── __init__.py
│   │   ├── weather_data.py       # 天气数据获取
│   │   └── preprocessor.py       # 数据预处理
│   ├── model/                    # 模型定义模块
│   │   ├── __init__.py
│   │   ├── base_model.py         # 基础模型类
│   │   ├── temperature_model.py  # 温度预测模型
│   │   └── rainfall_model.py     # 降雨预测模型
│   ├── trainer/                  # 训练模块
│   │   ├── __init__.py
│   │   ├── base_trainer.py       # 基础训练器
│   │   ├── temperature_trainer.py # 温度模型训练器
│   │   └── rainfall_trainer.py   # 降雨模型训练器
│   └── utils/                    # 工具模块
│       ├── __init__.py
│       ├── early_stopping.py    # 早停机制
│       ├── metrics.py            # 评估指标
│       └── visualization.py     # 可视化工具
├── config/                       # 配置文件
│   ├── __init__.py
│   └── config.py                 # 项目配置
├── tests/                        # 单元测试
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_models.py
│   └── test_trainers.py
├── models/                       # 训练好的模型
│   ├── 北京/
│   ├── 上海/
│   ├── 广州/
│   ├── 深圳/
│   └── 西安/
├── data/                         # 数据目录
├── logs/                         # 日志目录
├── plots/                        # 图表目录
├── train.py                      # 训练脚本入口
├── WeatherApp.py                 # Streamlit网页应用
├── requirements.txt              # 依赖包列表
└── README.md                     # 项目文档
```

## 🔧 安装与配置

### 环境要求

- Python 3.8+
- PyTorch 1.9.0+
- CUDA（可选，用于GPU加速）

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd Weather（修改版）

# 安装依赖
pip install -r requirements.txt
```

### 环境变量配置

```bash
# 设置环境（可选）
export WEATHER_ENV=development  # development/production/test
```

## 🚀 快速开始

### 1. 训练模型

```bash
# 训练温度预测模型
python train.py --city 西安 --model-type temperature --forecast-days 3

# 训练降雨回归模型
python train.py --city 西安 --model-type rainfall_regressor --forecast-days 3

# 训练降雨分类模型
python train.py --city 西安 --model-type rainfall_classifier --forecast-days 3

# 自定义参数训练
python train.py --city 北京 --model-type temperature --forecast-days 5 \
                --epochs 50 --batch-size 64 --learning-rate 0.001
```

### 2. 启动Web应用

```bash
streamlit run WeatherApp.py
```

### 3. 运行测试

```bash
# 运行所有测试
python -m pytest tests/

# 运行特定测试
python -m pytest tests/test_models.py -v
```

## 📚 详细使用说明

### 数据加载模块

```python
from src.data_loader import WeatherDataLoader, WeatherDataProcessor

# 加载数据
loader = WeatherDataLoader()
df = loader.fetch_weather_data("西安", days=365)

# 数据预处理
processor = WeatherDataProcessor()
processed_df = processor.preprocess_data(df)
```

### 模型创建

```python
from src.model import WeatherModelFactory

# 创建温度预测模型
model = WeatherModelFactory.create_model(
    model_type='temperature',
    input_dim=15,
    hidden_dim=64,
    forecast_days=3
)

# 创建降雨分类模型
model = WeatherModelFactory.create_model(
    model_type='rainfall_classifier',
    input_dim=15,
    hidden_dim=64,
    forecast_days=3,
    num_classes=5
)
```

### 模型训练

```python
from src.trainer import TemperatureTrainer

# 创建训练器
trainer = TemperatureTrainer(model)

# 训练模型
history = trainer.train(train_loader, val_loader, num_epochs=50)

# 评估模型
metrics = trainer.evaluate(test_loader)
```

### 配置管理

```python
from config import config

# 访问配置
print(config.CITIES)              # 支持的城市列表
print(config.HIDDEN_DIM)          # 隐藏层维度
print(config.RAINFALL_THRESHOLDS) # 降雨分类阈值

# 获取模型路径
model_path = config.get_model_path("西安", "temperature", 3)
```

## 🎯 模型架构

### 温度预测模型（MultiOutputTemperaturePredictor）

- **输入**：15维特征序列（7天历史数据）
- **架构**：双向LSTM + 多头注意力 + 三分支输出
- **输出**：最高温度、最低温度、平均温度（同时预测）
- **约束**：物理约束确保 高温 ≥ 平均温 ≥ 低温

### 降雨预测模型

#### 回归模型（SimpleRainfallPredictor）
- **输入**：15维特征序列
- **架构**：CNN特征提取 + Transformer编码器
- **输出**：连续降雨量值
- **后处理**：阈值过滤（< 5mm设为0）

#### 分类模型（RainfallClassifierPredictor）
- **输入**：15维特征序列
- **架构**：CNN + Transformer + 分类头
- **输出**：5个降雨等级（无降雨、小雨、中雨、大雨、暴雨）
- **阈值**：0-5mm, 5-10mm, 10-18mm, 18-38mm, 38-75mm

## 📊 评估指标

### 温度预测
- **回归指标**：MSE、MAE、RMSE、R²
- **物理约束**：温度逻辑一致性检查
- **分温度类型**：分别评估高/低/平均温度

### 降雨预测
- **回归指标**：MSE、MAE、RMSE、R²
- **分类指标**：准确率、精确率、召回率、F1分数
- **降雨特定**：虚警率、漏报率、雨日准确率

## 🔧 配置说明

### 主要配置项

```python
# 模型配置
HIDDEN_DIM = 64              # 隐藏层维度
NUM_LAYERS = 2               # 网络层数
DROPOUT = 0.3                # Dropout比例
SEQUENCE_LENGTH = 7          # 输入序列长度
FORECAST_DAYS_LIST = [3, 5]  # 支持的预测天数

# 训练配置
BATCH_SIZE = 32              # 批次大小
LEARNING_RATE = 0.001        # 学习率
MAX_EPOCHS = 100             # 最大训练轮数
EARLY_STOPPING_PATIENCE = 15 # 早停容忍度

# 降雨分类阈值
RAINFALL_THRESHOLDS = {
    0: (0, 5),      # 无降雨
    1: (5, 10),     # 小雨  
    2: (10, 18),    # 中雨
    3: (18, 38),    # 大雨
    4: (38, 75)     # 暴雨
}
```

### 环境配置

- **development**：开发环境，详细日志，较少训练轮数
- **production**：生产环境，精简日志，完整训练
- **test**：测试环境，最少数据和轮数

## 🧪 测试

### 测试覆盖

- **数据加载测试**：数据获取、预处理、序列创建
- **模型测试**：前向传播、参数初始化、保存加载
- **训练测试**：训练流程、损失计算、指标评估
- **工具测试**：早停、指标计算、可视化

### 运行测试

```bash
# 所有测试
python -m pytest tests/ -v

# 覆盖率报告
python -m pytest tests/ --cov=src --cov-report=html

# 特定模块测试
python -m pytest tests/test_models.py::TestTemperatureModel -v
```

## 📈 性能优化

### 数据层面
- **多源数据**：支持6+个天气数据源
- **智能缓存**：本地数据缓存机制
- **数据验证**：自动数据质量检查

### 模型层面
- **注意力机制**：自动关注重要时间步
- **双向LSTM**：捕获前后时间依赖
- **多任务学习**：温度多输出联合训练

### 训练层面
- **早停机制**：防止过拟合
- **学习率调度**：自适应学习率调整
- **梯度裁剪**：稳定训练过程

## 🎨 可视化功能

- **预测结果图**：真实vs预测对比
- **性能指标图**：模型评估可视化
- **训练历史图**：损失和指标变化
- **特征重要性图**：特征贡献分析

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🆘 问题排查

### 常见问题

1. **网络连接问题**
   - 系统会自动切换到备用数据源
   - 最终会使用高质量模拟数据

2. **内存不足**
   - 减少`BATCH_SIZE`和`DATA_DAYS`
   - 使用CPU训练：设置`CUDA_VISIBLE_DEVICES=""`

3. **模型不收敛**
   - 调整学习率和网络架构
   - 检查数据质量和特征工程

### 日志分析

```bash
# 查看训练日志
tail -f logs/training.log

# 查看错误日志
grep "ERROR" logs/training.log
```

## 📞 联系方式

- 项目维护者：[Your Name]
- 邮箱：[your.email@example.com]
- 项目地址：[https://github.com/username/weather-prediction-ai]

---

**感谢使用天气预测AI系统！🌟** 