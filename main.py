import cv2
import dlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from collections import defaultdict

# 初始化DLIB人脸检测器和关键点预测器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

def get_facial_landmarks(image_path):
    """提取人脸关键点坐标"""
    image = cv2.imread(str(image_path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        return None
        
    landmarks = predictor(gray, faces[0])
    return np.array([[p.x, p.y] for p in landmarks.parts()])

def calculate_facial_proportions(landmarks):
    """根据定义计算五官尺寸和位置"""
    # 下巴最低点作为原点 (索引8)
    origin = landmarks[8].copy()
    
    # 坐标系转换：原点移至下巴
    normalized_landmarks = landmarks - origin
    
    # 定义关键点索引 (基于68点模型)
    FACE_WIDTH = np.linalg.norm(landmarks[0] - landmarks[16])
    
    # 眼睛计算
    LEFT_EYE_WIDTH = np.linalg.norm(landmarks[36] - landmarks[39])
    LEFT_EYE_HEIGHT = np.linalg.norm(landmarks[37] - landmarks[41])
    LEFT_EYE_CENTER = np.mean([landmarks[36], landmarks[39]], axis=0) - origin
    
    # 类似计算右眼、眉毛、鼻子、嘴巴等...
    # 完整实现需要根据研究定义补充所有器官的计算逻辑
    
    return {
        'face_width': FACE_WIDTH,
        'left_eye_width': LEFT_EYE_WIDTH,
        'left_eye_height': LEFT_EYE_HEIGHT,
        'left_eye_ratio': LEFT_EYE_HEIGHT / LEFT_EYE_WIDTH,
        'left_eye_x': LEFT_EYE_CENTER[0],
        'left_eye_y': LEFT_EYE_CENTER[1],
        # 添加其他特征...
    }

def analyze_portraits(image_dir, max_images=500):
    """批量分析名画数据"""
    image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.JPG"))
    image_paths = image_paths[:max_images]

    print(image_paths)
    results = defaultdict(list)
    
    for path in image_paths:
        try:
            landmarks = get_facial_landmarks(path)
            if landmarks is not None:
                proportions = calculate_facial_proportions(landmarks)
                for key, value in proportions.items():
                    results[key].append(value)
        except Exception as e:
            print(f"Error processing {path}: {str(e)}")
    
    return pd.DataFrame(results)

def calculate_aesthetic_ranges(df):
    """计算美学数值范围"""
    aesthetic_data = {}
    
    # 尺寸和比例分析
    size_columns = [col for col in df.columns if 'width' in col or 'height' in col]
    ratio_columns = [col for col in df.columns if 'ratio' in col]
    
    for col in size_columns + ratio_columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        median = np.median(col_data)
        q1, q3 = np.percentile(col_data, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        aesthetic_data[col] = {
            'median': median,
            'aesthetic_range': (q1 - 1.5*iqr, q3 + 1.5*iqr)
        }
    
    # 位置分布分析 (使用核密度估计)
    position_columns = [col for col in df.columns if 'x' in col or 'y' in col]
    
    for col in position_columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            continue
        try:
            kernel = stats.gaussian_kde(col_data)
            x = np.linspace(col_data.min(), col_data.max(), 100)
            y = kernel(x)
            peak = x[y.argmax()]  # 最优位置
            
            # 计算85%置信区间作为美学范围
            cumulative = np.cumsum(y)
            cumulative /= cumulative.max()
            lower = x[np.argmax(cumulative >= 0.075)]
            upper = x[np.argmax(cumulative >= 0.925)]
            
            aesthetic_data[col] = {
                'optimal': peak,
                'aesthetic_range': (lower, upper)
            }
        except Exception:
            continue
    
    return aesthetic_data

def visualize_results(df, aesthetic_data):
    """可视化分析结果"""
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    
    # 示例：眼睛宽高比分布
    ratios = df['left_eye_ratio']
    axs[0].hist([x for x in ratios if pd.notnull(x)], bins=30)
    if 'left_eye_ratio' in aesthetic_data and 'median' in aesthetic_data['left_eye_ratio']:
        axs[0].axvline(aesthetic_data['left_eye_ratio']['median'], color='r', linestyle='dashed')
    axs[0].set_title('Left Eye Aspect Ratio Distribution')
    
    # 示例：眼睛位置分布
    x_data = df['left_eye_x']
    y_data = df['left_eye_y']
    axs[1].scatter(x_data, y_data, alpha=0.5)
    if ('left_eye_x' in aesthetic_data and 'optimal' in aesthetic_data['left_eye_x'] and
        'left_eye_y' in aesthetic_data and 'optimal' in aesthetic_data['left_eye_y']):
        axs[1].scatter(aesthetic_data['left_eye_x']['optimal'], aesthetic_data['left_eye_y']['optimal'], marker='*', s=200)
    axs[1].set_title('Left Eye Position Distribution')
    
    # 更多可视化...
    plt.tight_layout()
    plt.savefig('facial_aesthetics_analysis.png', dpi=300)

# 主流程
if __name__ == "__main__":
    # 步骤1: 数据采集
    print('analyze..')
    portrait_data = analyze_portraits("famous_portraits/")
    print(portrait_data)
    
    # 步骤2: 计算美学参数
    aesthetic_ranges = calculate_aesthetic_ranges(portrait_data)
    
    # 步骤3: 可视化结果
    visualize_results(portrait_data, aesthetic_ranges)
    
    # 输出最优值和范围
    print("面部美学参数:")
    for feature, values in aesthetic_ranges.items():
        if 'median' in values:
            print(f"{feature}: 最佳值={values['median']:.2f}, 美学范围=[{values['aesthetic_range'][0]:.2f}, {values['aesthetic_range'][1]:.2f}]")
        else:
            print(f"{feature}: 最优位置={values['optimal']:.2f}, 美学范围=[{values['aesthetic_range'][0]:.2f}, {values['aesthetic_range'][1]:.2f}]")

# 帮助新手评估的附加功能
def evaluate_user_drawing(landmarks, aesthetic_ranges):
    """评估用户画作的面部比例"""
    evaluation = {}
    proportions = calculate_facial_proportions(landmarks)
    
    for feature, value in proportions.items():
        if feature in aesthetic_ranges:
            range_min, range_max = aesthetic_ranges[feature]['aesthetic_range']
            is_within_range = range_min <= value <= range_max
            median = aesthetic_ranges[feature].get('median', 
                      aesthetic_ranges[feature].get('optimal'))
            deviation = abs(value - median) / median * 100
            
            evaluation[feature] = {
                'value': value,
                'within_range': is_within_range,
                'deviation': f"{deviation:.1f}%"
            }
    
    return evaluation


