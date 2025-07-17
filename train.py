"""
天气预测AI系统 - 训练脚本入口
模块化版本的训练脚本
"""
import sys
import os
import argparse
import logging
import torch
import numpy as np
from datetime import datetime
from tqdm import tqdm

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from src.data_loader import WeatherDataLoader, WeatherDataProcessor
from src.model import WeatherModelFactory
from src.trainer import TemperatureTrainer, RainfallTrainer

# 设置日志
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOG_DIR, 'training.log'))
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="天气预测AI系统训练脚本")
    
    parser.add_argument('--city', type=str, default=config.DEFAULT_CITY,
                       choices=config.CITIES, help='城市名称')
    parser.add_argument('--model-type', type=str, default='temperature',
                       choices=['temperature', 'rainfall_regressor', 'rainfall_classifier'],
                       help='模型类型')
    parser.add_argument('--forecast-days', type=int, default=3,
                       choices=config.FORECAST_DAYS_LIST, help='预测天数')
    parser.add_argument('--epochs', type=int, default=config.MAX_EPOCHS,
                       help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                       help='批次大小')
    parser.add_argument('--learning-rate', type=float, default=config.LEARNING_RATE,
                       help='学习率')
    parser.add_argument('--hidden-dim', type=int, default=config.HIDDEN_DIM,
                       help='隐藏层维度')
    parser.add_argument('--num-layers', type=int, default=config.NUM_LAYERS,
                       help='网络层数')
    parser.add_argument('--dropout', type=float, default=config.DROPOUT,
                       help='Dropout比例')
    parser.add_argument('--save-model', action='store_true', default=True,
                       help='是否保存模型')
    parser.add_argument('--eval-only', action='store_true',
                       help='仅评估，不训练')
    parser.add_argument('--data-days', type=int, default=config.DATA_DAYS,
                       help='使用的数据天数')
    parser.add_argument('--train-all', action='store_true',
                       help='训练所有城市的温度和降雨量的3天和5天模型')
    parser.add_argument('--continue-on-error', action='store_true', default=True,
                       help='遇到错误时继续训练其他模型')
    
    return parser.parse_args()


def prepare_data(city: str, data_days: int):
    """准备训练数据"""
    logger.info(f"正在为 {city} 准备数据...")
    
    # 加载数据
    data_loader = WeatherDataLoader()
    df = data_loader.fetch_weather_data(city, data_days)
    
    # 数据预处理
    processor = WeatherDataProcessor()
    if not processor.validate_data(df):
        raise ValueError("数据质量验证失败")
    
    processed_df = processor.preprocess_data(df, city=city)
    
    logger.info(f"数据准备完成，共 {len(processed_df)} 条记录")
    
    return processed_df, processor


def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size: int):
    """创建数据加载器"""
    from torch.utils.data import TensorDataset, DataLoader
    
    # 转换为PyTorch张量
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = None
    if len(X_val) > 0:
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    test_loader = None
    if len(X_test) > 0:
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


def train_all_models(args):
    """训练所有城市的所有模型组合"""
    
    # 定义要训练的所有组合
    cities = config.CITIES
    model_types = ['temperature', 'rainfall_regressor']
    forecast_days_list = config.FORECAST_DAYS_LIST
    
    # 计算总任务数
    total_tasks = len(cities) * len(model_types) * len(forecast_days_list)
    
    logger.info("=" * 80)
    logger.info("开始训练所有城市的所有模型")
    logger.info(f"城市: {cities}")
    logger.info(f"模型类型: {model_types}")
    logger.info(f"预测天数: {forecast_days_list}")
    logger.info(f"总任务数: {total_tasks}")
    logger.info("=" * 80)
    
    # 训练统计
    completed_tasks = 0
    failed_tasks = []
    successful_tasks = []
    
    # 使用进度条
    with tqdm(total=total_tasks, desc="训练进度", unit="task") as pbar:
        
        for city in cities:
            for model_type in model_types:
                for forecast_days in forecast_days_list:
                    
                    task_name = f"{city}_{model_type}_{forecast_days}days"
                    
                    try:
                        logger.info(f"\n开始训练任务: {task_name}")
                        logger.info(f"进度: {completed_tasks + 1}/{total_tasks}")
                        
                        # 创建任务专用的参数副本
                        task_args = argparse.Namespace(**vars(args))
                        task_args.city = city
                        task_args.model_type = model_type
                        task_args.forecast_days = forecast_days
                        task_args.eval_only = False  # 确保进行训练
                        
                        # 训练单个模型
                        train_single_model(task_args)
                        
                        successful_tasks.append(task_name)
                        logger.info(f"✅ 任务 {task_name} 训练成功")
                        
                    except Exception as e:
                        error_msg = f"❌ 任务 {task_name} 训练失败: {str(e)}"
                        logger.error(error_msg)
                        failed_tasks.append((task_name, str(e)))
                        
                        if not args.continue_on_error:
                            logger.error("遇到错误，停止训练")
                            raise
                    
                    finally:
                        completed_tasks += 1
                        pbar.update(1)
                        pbar.set_postfix({
                            'Current': task_name,
                            'Success': len(successful_tasks),
                            'Failed': len(failed_tasks)
                        })
    
    # 打印最终统计
    logger.info("\n" + "=" * 80)
    logger.info("训练完成！最终统计:")
    logger.info(f"总任务数: {total_tasks}")
    logger.info(f"成功完成: {len(successful_tasks)}")
    logger.info(f"失败任务: {len(failed_tasks)}")
    logger.info(f"成功率: {len(successful_tasks)/total_tasks*100:.1f}%")
    
    if successful_tasks:
        logger.info("\n✅ 成功完成的任务:")
        for task in successful_tasks:
            logger.info(f"  - {task}")
    
    if failed_tasks:
        logger.info("\n❌ 失败的任务:")
        for task_name, error in failed_tasks:
            logger.info(f"  - {task_name}: {error}")
    
    logger.info("=" * 80)
    
    return {
        'total': total_tasks,
        'successful': len(successful_tasks),
        'failed': len(failed_tasks),
        'successful_tasks': successful_tasks,
        'failed_tasks': failed_tasks
    }


def train_single_model(args):
    """训练单个模型（重构原来的train_model函数）"""
    logger.info(f"开始训练 {args.model_type} 模型，城市: {args.city}，预测天数: {args.forecast_days}")
    
    # 创建必要目录
    config.create_directories()
    
    # 为当前城市创建模型目录
    city_model_dir = os.path.join(config.MODEL_DIR, args.city)
    os.makedirs(city_model_dir, exist_ok=True)
    
    # 准备数据
    df, processor = prepare_data(args.city, args.data_days)
    feature_cols = processor.get_feature_columns()
    
    # 根据模型类型准备数据
    if args.model_type == 'temperature':
        target_cols = ['high_temp', 'low_temp', 'avg_temp']
        X, y = processor.create_sequences(df, feature_cols, target_cols, 
                                        config.SEQUENCE_LENGTH, args.forecast_days)
    elif args.model_type == 'rainfall_classifier':
        target_col = 'rain'
        X, y = processor.create_sequences_for_classification(df, feature_cols, target_col,
                                                           config.SEQUENCE_LENGTH, args.forecast_days)
    else:  # rainfall_regressor
        target_col = 'rain'
        X, y = processor.create_sequences(df, feature_cols, target_col,
                                        config.SEQUENCE_LENGTH, args.forecast_days)
    
    # 数据划分
    X_train, X_val, X_test, y_train, y_val, y_test = processor.split_train_val_test_by_time(
        X, y, df, config.SEQUENCE_LENGTH, args.forecast_days
    )
    
    # 数据标准化
    X_train_scaled, X_val_scaled, X_test_scaled = processor.scale_data(X_train, X_val, X_test)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train_scaled, y_train, X_val_scaled, y_val, X_test_scaled, y_test, args.batch_size
    )
    
    # 创建模型
    input_dim = X_train_scaled.shape[-1]
    model_kwargs = {
        'input_dim': input_dim,
        'hidden_dim': args.hidden_dim,
        'forecast_days': args.forecast_days,
        'num_layers': args.num_layers,
        'dropout': args.dropout
    }
    
    if args.model_type == 'rainfall_classifier':
        model_kwargs['num_classes'] = config.RAINFALL_CLASSES
    
    model = WeatherModelFactory.create_model(args.model_type, **model_kwargs)
    
    # 创建训练器
    if args.model_type == 'temperature':
        trainer = TemperatureTrainer(model, config)
    else:
        trainer = RainfallTrainer(model, config, model_type=args.model_type)
    
    # 更新训练配置
    trainer.config.MAX_EPOCHS = args.epochs
    trainer.config.LEARNING_RATE = args.learning_rate
    trainer.config.BATCH_SIZE = args.batch_size
    
    if not args.eval_only:
        # 训练模型
        history = trainer.train(train_loader, val_loader, args.epochs)
        
        # 保存模型
        if args.save_model:
            model_path = config.get_model_path(args.city, args.model_type, args.forecast_days)
            trainer.save_model(model_path, {
                'city': args.city,
                'model_type': args.model_type,
                'forecast_days': args.forecast_days,
                'training_args': vars(args),
                'feature_columns': feature_cols,
                'training_date': datetime.now().isoformat()
            })
            logger.info(f"模型已保存到: {model_path}")
    
    # 评估模型
    if test_loader is not None:
        logger.info("开始模型评估...")
        metrics = trainer.evaluate(test_loader)
        
        # 打印评估结果
        logger.info("评估结果:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")
            elif isinstance(value, dict):
                logger.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        logger.info(f"    {sub_key}: {sub_value:.4f}")
    
    logger.info(f"任务 {args.city}_{args.model_type}_{args.forecast_days}days 完成！")


def train_model(args):
    """训练模型的入口函数（保持向后兼容）"""
    return train_single_model(args)


def main():
    """主函数"""
    print("天气预测AI系统 - 模块化训练脚本")
    print("=" * 50)
    
    # 解析参数
    args = parse_arguments()
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 检查是否训练所有模型
    if args.train_all:
        logger.info("启动批量训练模式 - 训练所有城市的所有模型")
        
        # 打印批量训练配置
        logger.info(f"批量训练配置:")
        logger.info(f"  城市: {config.CITIES}")
        logger.info(f"  模型类型: ['temperature', 'rainfall_regressor']")
        logger.info(f"  预测天数: {config.FORECAST_DAYS_LIST}")
        logger.info(f"  训练轮数: {args.epochs}")
        logger.info(f"  批次大小: {args.batch_size}")
        logger.info(f"  学习率: {args.learning_rate}")
        logger.info(f"  继续训练遇错误: {args.continue_on_error}")
        
        # 开始批量训练
        results = train_all_models(args)
        
        # 根据结果退出
        if results['failed'] > 0 and not args.continue_on_error:
            sys.exit(1)
        
    else:
        logger.info("启动单模型训练模式")
        
        # 打印单模型训练配置
        logger.info(f"单模型训练配置:")
        logger.info(f"  城市: {args.city}")
        logger.info(f"  模型类型: {args.model_type}")
        logger.info(f"  预测天数: {args.forecast_days}")
        logger.info(f"  训练轮数: {args.epochs}")
        logger.info(f"  批次大小: {args.batch_size}")
        logger.info(f"  学习率: {args.learning_rate}")
        logger.info(f"  隐藏维度: {args.hidden_dim}")
        logger.info(f"  网络层数: {args.num_layers}")
        logger.info(f"  Dropout: {args.dropout}")
        
        # 开始单模型训练
        train_model(args)


if __name__ == "__main__":
    main() 