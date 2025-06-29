import re
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import json

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体作为默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

COLUMNS = [
    '省份', '流域', '断面名称', '监测时间', '水质类别',
    '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
    '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
    '总磷(mg/L)', '总氮(mg/L)', '叶绿素α(mg/L)',
    '藻密度(cells/L)', '站点情况'
]

# 数值型字段列表（与COLUMNS对应）
NUMERIC_COLS = [
    '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
    '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
    '总磷(mg/L)', '总氮(mg/L)'
]


def parse_data_file(file_path: Path):
    """通用数据解析入口"""
    if file_path.suffix == '.json':
        return parse_temporal_json(file_path)
    elif file_path.suffix == '.csv':
        return parse_spatial_csv(file_path)
    return pd.DataFrame()


def parse_temporal_json(json_file: Path):
    """强化版JSON解析"""
    try:
        # 从路径解析年月（支持多级目录）
        year_month = "-".join(json_file.parts[-2].split("-")[:2])  # 适配多级目录结构

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 强化字段清洗（处理所有HTML标签）
        columns = [re.sub(r'<.*?>', '', col).strip() for col in data['thead']]

        # 强制字段映射
        column_mapping = {
            '断面名称': ['监测点名称', 'site', 'Site Name'],
            '监测时间': ['时间', 'Timestamp']
        }

        # 自动匹配列名
        final_cols = {}
        for target, aliases in column_mapping.items():
            for idx, col in enumerate(columns):
                if col in aliases:
                    final_cols[target] = idx
                    break
            else:
                raise ValueError(f"缺失关键列: {target}")

        # 数据解析
        records = []
        for row in data['tbody']:
            record = {}
            for target, idx in final_cols.items():
                raw_value = row[idx] if idx < len(row) else None

                # 深度清洗HTML内容
                if isinstance(raw_value, str):
                    cleaned = re.sub(r'<.*?>', '', raw_value).strip()
                    cleaned = cleaned.split('(')[0].strip()  # 移除括号内容
                else:
                    cleaned = raw_value

                record[target] = cleaned

            # 数值字段处理
            for idx, col in enumerate(columns):
                if col not in final_cols.values():
                    record[col] = row[idx] if idx < len(row) else None

            records.append(record)

        df = pd.DataFrame(records)

        # 强化生成监测点字段的逻辑（四重保障）
        df['数据来源_监测点'] = (
                df.get('断面名称')
                or df.get('监测点名称')
                or df.get('site')
                or f"JSON_{json_file.stem}"
        )

        # 添加字段存在性断言
        if '数据来源_监测点' not in df.columns:
            raise ValueError("监测点字段生成失败")

        # 时间处理（自动补全年份）
        df['监测时间'] = pd.to_datetime(
            df['监测时间'].apply(lambda x: f"{year_month}-{x}" if x else None),
            format='%Y-%m-%d %H:%M',
            errors='coerce'
        )

        return df.dropna(subset=['数据来源_监测点', '监测时间'])


    except Exception as e:
        print(f"JSON解析失败 {json_file}: {str(e)}")
        return pd.DataFrame()


def parse_spatial_csv(csv_file: Path):
    """修复版CSV解析"""
    try:
        # 初始化DataFrame保障
        df = pd.DataFrame()

        # 解析地理路径（增强容错）
        try:
            path_parts = csv_file.parts
            province_idx = path_parts.index("water_quality_by_name") + 1
            province = path_parts[province_idx]
            basin = path_parts[province_idx + 1]
            section = path_parts[province_idx + 2]
        except (ValueError, IndexError):
            province = csv_file.parent.parent.name
            basin = csv_file.parent.name
            section = csv_file.stem

        # 读取CSV文件（增强编码检测）
        encodings = ['utf-8', 'gbk', 'gb18030', 'big5']
        for enc in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            return pd.DataFrame()

        # 强制生成监测点字段
        section_clean = re.sub(r'[\(\)（）]', '', section).strip()
        df['数据来源_监测点'] = f"{province}-{basin}-{section_clean}"

        # 时间处理增强
        if '监测时间' in df.columns:
            df['监测时间'] = pd.to_datetime(df['监测时间'], errors='coerce')
        else:
            df['监测时间'] = pd.Timestamp.now()

        return df

    except Exception as e:
        print(f"CSV解析失败 {csv_file}: {type(e).__name__}-{str(e)}")
        return pd.DataFrame()  # 确保返回空DataFrame


def parse_json_to_dataframe(json_file: Path):
    """专门处理JSON格式的水质数据"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 清洗列名
        columns = [col.split('<')[0].strip() for col in data['thead']]
        columns = [
            '省份', '流域', '断面名称', '监测时间', '水质类别',
            '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
            '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
            '总磷(mg/L)', '总氮(mg/L)', '叶绿素α(mg/L)', '藻密度(cells/L)', '站点情况'
        ]

        # 解析数据
        records = []
        for row in data['tbody']:
            record = {}
            for i, col in enumerate(columns):
                raw_value = row[i]

                # 处理HTML标签和特殊值
                if isinstance(raw_value, str) and 'data-toggle' in raw_value:
                    try:
                        # 提取原始值
                        value = raw_value.split("'")[3]
                    except:
                        value = raw_value.split('>')[1].split('<')[0]
                else:
                    value = raw_value

                # 特殊值处理
                if value in ('--', '*', 'null'):
                    value = None

                record[col] = value
            records.append(record)

        df = pd.DataFrame(records)

        # 生成监测点字段
        df['数据来源_监测点'] = df['断面名称'].str.split('<').str[0].str.strip()

        # 时间处理（补充年份信息）
        df['监测时间'] = '2020-' + df['监测时间'].str.replace(' ', ' ')
        df['监测时间'] = pd.to_datetime(df['监测时间'], format='%Y-%m-%d %H:%M', errors='coerce')

        # 数值类型转换
        numeric_cols = [
            '水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
            '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
            '总磷(mg/L)', '总氮(mg/L)', '叶绿素α(mg/L)', '藻密度(cells/L)'
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df.dropna(subset=['监测时间'])

    except Exception as e:
        print(f"JSON解析失败 {json_file.name}: {str(e)}")
        return pd.DataFrame()

def parse_custom_date(date_str):
    """支持更多日期格式的解析"""
    # 处理空值和异常类型
    if pd.isna(date_str) or not isinstance(date_str, (str, int, float)):
        return pd.NaT

    # 统一转换为字符串处理
    date_str = str(date_str).strip()

    # 定义所有可能的日期格式
    date_formats = [
        '%Y/%m/%d %H:%M',  # 原始数据格式
        '%Y-%m-%d %H:%M:%S',  # ISO标准格式
        '%Y%m%d',  # 纯数字格式
        '%d-%b-%y %H:%M',  # 01-Jan-25 04:00
        '%Y年%m月%d日%H时'  # 中文格式
    ]

    # 尝试所有格式
    for fmt in date_formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue

    # 最后尝试自动推断
    try:
        return pd.to_datetime(date_str)
    except:
        return pd.NaT


def parse_csv_to_dataframe(csv_file, root_dir):
    try:
        # 动态检测编码格式
        encodings = ['utf-8', 'gbk', 'gb2312']
        for enc in encodings:
            try:
                df = pd.read_csv(
                    csv_file,
                    encoding=enc,
                    engine='python',
                    dtype={'监测时间': 'str'},
                    na_values=['*', '--', 'NA', 'null', ''],
                    on_bad_lines='warn'
                )
                break
            except UnicodeDecodeError:
                continue

        # 动态列名映射（支持中英文混合）
        name_mappings = {
            '数据来源_监测点': ['断面名称', 'Monitoring Site', 'site_name', 'Site', '监测点名称'],
            '监测时间': ['时间', 'Timestamp', '监测时间', 'datetime']
        }

        # 自动匹配列名
        for target_col, possible_names in name_mappings.items():
            matched_cols = [col for col in df.columns if col in possible_names]
            if matched_cols:
                df.rename(columns={matched_cols[0]: target_col}, inplace=True)
            else:
                # 从文件路径生成监测点名称
                if target_col == '数据来源_监测点':
                    path_parts = csv_file.relative_to(root_dir).parts
                    df[target_col] = f"{path_parts[-3]}-{path_parts[-2]}"  # 省份-流域

        # 强制创建必要字段
        if '数据来源_监测点' not in df.columns:
            df['数据来源_监测点'] = csv_file.stem  # 使用文件名作为监测点名称

        # 日期解析（增强容错）
        df['监测时间'] = df['监测时间'].apply(lambda x: parse_custom_date(x))
        return df.dropna(subset=['监测时间'])

    except Exception as e:
        print(f"文件解析失败 {csv_file.name}: {str(e)}")
        return pd.DataFrame()


def load_all_water_data(root_dir: str):
    """终极数据加载（增强异常处理）"""
    try:
        root_path = Path(root_dir)
        all_dfs = []

        # 检查根目录是否存在
        if not root_path.exists():
            print(f"错误：数据目录不存在 {root_path}")
            return pd.DataFrame()

        # 加载时间数据（JSON）添加目录存在检查
        temporal_root = root_path
        if temporal_root.exists():
            for json_file in temporal_root.rglob("*.json"):
                df = parse_temporal_json(json_file)
                if not df.empty:
                    all_dfs.append(df)
        else:
            print(f"警告：未找到时间序列数据目录 {temporal_root}")

        # 加载空间数据（CSV）添加目录存在检查
        spatial_root = root_path / "water_quality_by_name"
        if spatial_root.exists():
            for csv_file in spatial_root.rglob("*.csv"):
                df = parse_spatial_csv(csv_file)
                if not df.empty:
                    all_dfs.append(df)
        else:
            print(f"警告：未找到地理分布数据目录 {spatial_root}")

        # 合并数据集
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True)

            # 字段保障（三重验证）
            required_col = '数据来源_监测点'
            backup_cols = ['断面名称', 'Monitoring Site', 'site_name']

            if required_col not in combined.columns:
                for col in backup_cols:
                    if col in combined.columns:
                        combined[required_col] = combined[col]
                        break
                else:
                    combined[required_col] = "默认监测点"

            return combined.fillna({required_col: "未知监测点"})

        return pd.DataFrame()

    except Exception as e:
        print(f"数据加载严重错误: {str(e)}")
        return pd.DataFrame()


def validate_dataset(df):
    """数据质量校验"""
    # 检查时间连续性
    time_diff = df.groupby('断面名称')['监测时间'].diff().dt.total_seconds()
    time_anomalies = df[time_diff > 4 * 3600]  # 超过4小时间隔

    # 数值范围校验
    value_ranges = {
        '水温(℃)': (-5, 40),
        'pH(无量纲)': (6, 9),
        '溶解氧(mg/L)': (0, 20)
    }

    for col, (min_val, max_val) in value_ranges.items():
        outliers = df[(df[col] < min_val) | (df[col] > max_val)]
        if not outliers.empty:
            print(f"{col} 异常值数量：{len(outliers)}")

    return df

def plot_water_quality(data, site_name):
    """增强型可视化函数"""
    site_data = data[data['断面名称'] == site_name].sort_values('监测时间')

    # 创建带异常值标注的交互式图表
    fig = px.line(site_data, x='监测时间', y=NUMERIC_COLS,
                  title=f'{site_name} 水质参数趋势',
                  hover_data=['站点情况'],
                  facet_col='variable', facet_col_wrap=3,
                  labels={'value': '测量值', 'variable': '参数'},
                  height=900)

    # 标记异常数据点
    anomalies = site_data[site_data['站点情况'] == '维护']
    if not anomalies.empty:
        fig.add_scatter(x=anomalies['监测时间'],
                        y=[0] * len(anomalies),  # 底部标记
                        mode='markers',
                        marker=dict(color='red', size=8),
                        name='维护时段')

    fig.show()


def interactive_visualization(data):
    """
    交互式可视化界面
    """
    # 获取所有监测点名称
    sites = data['数据来源_监测点'].unique()

    print("\n可用的监测点:")
    for i, site in enumerate(sites, 1):
        print(f"{i}. {site}")

    while True:
        try:
            choice = input("\n请输入要查看的监测点编号(输入q退出): ")
            if choice.lower() == 'q':
                break

            site_idx = int(choice) - 1
            if 0 <= site_idx < len(sites):
                selected_site = sites[site_idx]

                # 获取该监测点可用的参数
                site_data = data[data['数据来源_监测点'] == selected_site]
                available_params = [col for col in site_data.columns
                                    if col in ['水温(℃)', 'pH(无量纲)', '溶解氧(mg/L)', '电导率(μS/cm)',
                                               '浊度(NTU)', '高锰酸盐指数(mg/L)', '氨氮(mg/L)',
                                               '总磷(mg/L)', '总氮(mg/L)', '叶绿素α(mg/L)', '藻密度(cells/L)']]

                print(f"\n监测点 '{selected_site}' 可用的参数:")
                for i, param in enumerate(available_params, 1):
                    print(f"{i}. {param}")

                param_choice = input("请输入要绘制的参数编号(逗号分隔,留空绘制全部): ")
                if param_choice.strip():
                    try:
                        param_indices = [int(x) - 1 for x in param_choice.split(',')]
                        selected_params = [available_params[i] for i in param_indices
                                           if 0 <= i < len(available_params)]
                    except:
                        print("输入无效，将绘制全部参数")
                        selected_params = None
                else:
                    selected_params = None

                # 绘制图表
                plot_water_quality(data, selected_site, selected_params)
            else:
                print("编号超出范围，请重新输入")

        except ValueError:
            print("请输入有效的编号")


# 主程序
if __name__ == "__main__":
    # 加载数据
    root_directory = '软件工程大作业数据/水质数据/water_quality_by_name'
    print("正在加载数据，请稍候...")
    all_water_data = load_all_water_data(root_directory)

    if not all_water_data.empty:
        print("\n数据加载完成!")
        print(f"总数据量: {len(all_water_data)} 条记录")
        print(f"监测点数量: {all_water_data['数据来源_监测点'].nunique()}")
        # print(f"时间范围: {all_water_data['监测时间'].min().date()} 至 {all_water_data['监测时间'].max().date()}")

        # 启动交互式可视化
        interactive_visualization(all_water_data)
    else:
        print("未找到任何CSV文件，请检查目录路径。")