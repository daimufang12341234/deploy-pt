from ultralytics import YOLO
import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import csv
from datetime import datetime

class MultiModelDetector:
    def __init__(self, model_configs):
        """
        初始化多模型检测器
        Args:
            model_configs: 模型配置列表，包含每个模型的路径和类别信息
        """
        self.models = []
        for config in model_configs:
            model = YOLO(config['model_path'])
            if config.get('classes'):
                print(f"\nCDJ Model Info:")
                print(f"Model path: {config['model_path']}")
                print(f"Available classes: {model.names}")  # 打印模型支持的类别
                print(f"Class mapping: {dict(enumerate(model.names))}")  # 打印类别索引映射
            
            model_dict = {
                'model': model,
                'class_name': config['class_name'],
                'point_colors': config['point_colors'],
                'classes': config.get('classes', None)
            }
            self.models.append(model_dict)
            
        self.target_size = (530, 802)  # (height, width)

    def add_black_border(self, image):
        """
        添加黑色边框使图片达到目标尺寸
        """
        h, w = image.shape[:2]
        padded_img = np.zeros((*self.target_size, 3), dtype=np.uint8)
        
        # 计算居中放置的位置
        x_offset = (self.target_size[1] - w) // 2
        y_offset = (self.target_size[0] - h) // 2
        
        # 将原图放在中心位置
        padded_img[y_offset:y_offset+h, x_offset:x_offset+w] = image
        
        return padded_img, (x_offset, y_offset)

    def detect(self, image):
        """
        执行多模型检测，使用框和关键点的综合置信度来选择最佳结果
        """
        padded_img, (x_offset, y_offset) = self.add_black_border(image)
        
        best_detection = None
        highest_confidence = -1
        
        # 调整阈值
        BOX_CONF_THRESH = 0.3  # 降低框置信度阈值
        KEYPOINT_CONF_THRESH = 0.5  # 降低关键点置信度阈值
        BOX_WEIGHT = 0.7
        KEYPOINT_WEIGHT = 0.3
        
        # 先处理CDJ多类别模型，再处理A模型
        for model_info in sorted(self.models, key=lambda x: x.get('classes') is not None, reverse=True):
            results = model_info['model'].predict(padded_img, conf=BOX_CONF_THRESH, verbose=False)
            result = results[0]
            
            if len(result.boxes) > 0:
                if model_info.get('classes'):
                    print(f"\nProcessing CDJ model")
                    print(f"Raw predictions: {result.boxes.cls.tolist()}")  # 打印原始类别预测
                    print(f"Raw confidences: {result.boxes.conf.tolist()}")  # 打印原始置信度
                    boxes = result.boxes
                    for box in boxes:
                        box_confidence = box.conf[0].item()
                        class_idx = int(box.cls[0].item())
                        class_name = model_info['classes'][class_idx]
                        print(f"Class: {class_name} (idx: {class_idx}), Confidence: {box_confidence:.4f}")
                        
                        if hasattr(result.keypoints, 'xy'):
                            box_idx = (boxes.conf == box.conf[0]).nonzero(as_tuple=True)[0][0]
                            if box_idx < len(result.keypoints.xy):
                                keypoints = result.keypoints.xy[box_idx].tolist()
                                keypoint_confs = result.keypoints.conf[box_idx].tolist() if hasattr(result.keypoints, 'conf') else [1.0] * len(keypoints)
                                print(f"Keypoint confidences: {[f'{conf:.4f}' for conf in keypoint_confs]}")
                                
                                if all(conf > KEYPOINT_CONF_THRESH for conf in keypoint_confs):
                                    avg_keypoint_conf = sum(keypoint_confs) / len(keypoint_confs)
                                    final_confidence = box_confidence * BOX_WEIGHT + avg_keypoint_conf * KEYPOINT_WEIGHT
                                    print(f"Final confidence: {final_confidence:.4f}")
                                    
                                    if final_confidence > highest_confidence:
                                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                                        highest_confidence = final_confidence
                                        best_detection = {
                                            'class_name': class_name,
                                            'box': [x1, y1, x2, y2],
                                            'box_confidence': box_confidence,
                                            'keypoint_confidences': keypoint_confs,
                                            'avg_keypoint_conf': avg_keypoint_conf,
                                            'final_confidence': final_confidence,
                                            'keypoints': keypoints,
                                            'point_colors': model_info['point_colors'][:len(keypoints)]
                                        }
                            else:
                                print("Rejected due to low keypoint confidence")
                        else:
                            print("No keypoints found")
                else:  # A模型
                    max_conf_idx = result.boxes.conf.argmax()
                    box = result.boxes[max_conf_idx]
                    box_confidence = box.conf[0].item()
                    
                    if hasattr(result.keypoints, 'xy') and len(result.keypoints.xy) > 0:
                        keypoints = result.keypoints.xy[0].tolist()
                        keypoint_confs = result.keypoints.conf[0].tolist() if hasattr(result.keypoints, 'conf') else [1.0] * len(keypoints)
                        
                        if all(conf > KEYPOINT_CONF_THRESH for conf in keypoint_confs):
                            avg_keypoint_conf = sum(keypoint_confs) / len(keypoint_confs)
                            final_confidence = box_confidence * BOX_WEIGHT + avg_keypoint_conf * KEYPOINT_WEIGHT
                            
                            if final_confidence > highest_confidence:
                                x1, y1, x2, y2 = box.xyxy[0].tolist()
                                highest_confidence = final_confidence
                                best_detection = {
                                    'class_name': model_info['class_name'],
                                    'box': [x1, y1, x2, y2],
                                    'box_confidence': box_confidence,
                                    'keypoint_confidences': keypoint_confs,
                                    'avg_keypoint_conf': avg_keypoint_conf,
                                    'final_confidence': final_confidence,
                                    'keypoints': keypoints,
                                    'point_colors': model_info['point_colors'][:len(keypoints)]
                                }
        
        return padded_img, [best_detection] if best_detection else []

    def visualize(self, image, detections):
        """
        可视化检测结果，显示综合置信度和交点
        """
        result_image = image.copy()
        
        for det in detections:
            # 绘制边界框
            x1, y1, x2, y2 = map(int, det['box'])
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 添加类别和置信度标签
            label = f"{det['class_name']} Box:{det['box_confidence']:.2f} Final:{det['final_confidence']:.2f}"
            cv2.putText(result_image, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 绘制关键点和编号
            keypoints = det['keypoints']
            colors = det['point_colors']
            class_name = det['class_name']
            
            # 绘制关键点、编号和置信度
            for i, ((x, y), color, conf) in enumerate(zip(keypoints, colors, det['keypoint_confidences']), 1):
                x, y = int(x), int(y)
                cv2.circle(result_image, (x, y), 4, color, -1)
                if class_name == 'A':
                    label = f"{i} ({conf:.2f})"
                else:
                    label = f"{class_name}_{i} ({conf:.2f})"
                cv2.putText(result_image, label, (x+5, y+5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # 如果是A类别，计算并绘制交点
            if class_name == 'A' and len(keypoints) == 4:
                intersection = calculate_intersection(
                    keypoints[0], keypoints[1],  # 1,2点
                    keypoints[2], keypoints[3]   # 3,4点
                )
                if intersection:
                    x, y = map(int, intersection)
                    # 绘制交点（黄点）
                    cv2.circle(result_image, (x, y), 4, (0, 255, 255), -1)
                    # 添加交点坐标标签
                    label = f"({x}, {y})"
                    cv2.putText(result_image, label, (x+10, y+10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    # 绘制延长线（虚线）
                    pt1 = tuple(map(int, keypoints[0]))
                    pt2 = tuple(map(int, keypoints[1]))
                    pt3 = tuple(map(int, keypoints[2]))
                    pt4 = tuple(map(int, keypoints[3]))
                    
                    # 使用虚线绘制延长线
                    cv2.line(result_image, pt1, pt2, (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.line(result_image, pt3, pt4, (255, 255, 0), 1, cv2.LINE_AA)
                    
                    # 延长线到交点
                    cv2.line(result_image, pt2, (x, y), (255, 255, 0), 1, cv2.LINE_AA)
                    cv2.line(result_image, pt4, (x, y), (255, 255, 0), 1, cv2.LINE_AA)
        
        return result_image

def process_folder(model_configs, input_folder, output_folder):
    """
    处理文件夹中的所有图片，并保存关键点坐标到CSV文件
    """
    # 初始化多模型检测器
    detector = MultiModelDetector(model_configs)
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 为每个类别创建结果存储字典
    results_dict = {}
    for config in model_configs:
        if config.get('classes'):  # 对于CDJ合并模型
            for class_name in config['classes']:
                results_dict[class_name] = []
        else:  # 对于A模型
            results_dict[config['class_name']] = []
    
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    
    # 获取所有图片文件
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(image_extensions)]
    image_files.sort(key=natural_sort_key)
    
    print(f"\n找到 {len(image_files)} 个图片文件")
    
    # 使用tqdm创建进度条
    for file_name in tqdm(image_files, desc="处理进度"):
        input_path = os.path.join(input_folder, file_name)
        output_path = os.path.join(output_folder, f'result_{file_name}')
        
        try:
            # 读取原始图像
            image = cv2.imread(input_path)
            if image is None:
                continue
            
            # 执行多模型检测（内部会自动添加黑边）
            padded_img, detections = detector.detect(image)
            
            # 保存检测结果
            if detections:
                for det in detections:  # 遍历所有检测结果
                    class_name = det['class_name']
                    
                    # 对D和JB类别使用更低的阈值
                    if class_name in ['D', 'JB']:
                        BOX_CONF_THRESH = 0.2
                        KEYPOINT_CONF_THRESH = 0.3
                    
                    # 准备结果数据
                    result_row = [file_name, f"{det['box_confidence']:.4f}"]
                    keypoints = det['keypoints']
                    
                    # 添加关键点坐标
                    for kpt in keypoints:
                        result_row.extend([f"{kpt[0]:.1f}", f"{kpt[1]:.1f}"])
                    
                    # 如果是A类别，计算交点
                    if class_name == 'A' and len(keypoints) == 4:
                        intersection = calculate_intersection(
                            keypoints[0], keypoints[1],  # 1,2点
                            keypoints[2], keypoints[3]   # 3,4点
                        )
                        if intersection:
                            result_row.extend([f"{intersection[0]:.1f}", f"{intersection[1]:.1f}"])
                        else:
                            result_row.extend(['NA', 'NA'])  # 如果没有交点
                    
                    # 添加到对应类别的结果列表
                    results_dict[class_name].append(result_row)
            
            # 可视化结果
            result_image = detector.visualize(padded_img, detections)
            
            # 保存结果图像
            cv2.imwrite(output_path, result_image)
            
        except Exception as e:
            print(f"处理 {file_name} 时出错: {str(e)}")
            continue
    
    # 保存排序后的结果到CSV文件
    for class_name, results in results_dict.items():
        if results:
            results.sort(key=lambda x: natural_sort_key(x[0]))
            csv_path = os.path.join(output_folder, f'{class_name}_keypoints.csv')
            with open(csv_path, 'w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                
                # 写入CSV头部
                header = ['image_name', 'confidence']
                num_points = len(results[0]) - 2  # 减去图片名和置信度列
                if class_name == 'A':
                    num_points = (num_points - 2) // 2  # 减去交点坐标，除以2得到点数
                else:
                    num_points = num_points // 2  # 除以2得到点数
                    
                for i in range(num_points):
                    point_label = str(i + 1) if class_name == 'A' else f'{class_name}_{i + 1}'
                    header.extend([f'{point_label}_x', f'{point_label}_y'])
                
                # 如果是A类别，添加交点坐标的列
                if class_name == 'A':
                    header.extend(['intersection_x', 'intersection_y'])
                
                csv_writer.writerow(header)
                
                # 写入排序后的结果
                for row in results:
                    csv_writer.writerow(row)

def natural_sort_key(s):
    """
    用于实现自然排序的键函数
    将字符串中的数字部分转换为整数进行比较
    """
    import re
    def atoi(text):
        return int(text) if text.isdigit() else text
    return [atoi(c) for c in re.split(r'(\d+)', s)]

def calculate_intersection(p1, p2, p3, p4):
    """
    计算两条直线的交点
    p1, p2: 第一条直线的两个点 [(x1,y1), (x2,y2)]
    p3, p4: 第二条直线的两个点 [(x3,y3), (x4,y4)]
    返回交点坐标 (x, y)
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    
    # 计算两条直线的斜率和截距
    # 处理垂直线的情况
    if x2 - x1 == 0:  # 第一条线垂直
        k1 = None
        b1 = None
        x = x1
        if x4 - x3 == 0:  # 两条线都垂直
            return None
        k2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - k2 * x3
        y = k2 * x + b2
    elif x4 - x3 == 0:  # 第二条线垂直
        k2 = None
        b2 = None
        x = x3
        k1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - k1 * x1
        y = k1 * x + b1
    else:  # 两条线都不垂直
        k1 = (y2 - y1) / (x2 - x1)
        b1 = y1 - k1 * x1
        k2 = (y4 - y3) / (x4 - x3)
        b2 = y3 - k2 * x3
        
        # 如果两线平行，返回None
        if k1 == k2:
            return None
            
        # 计算交点
        x = (b2 - b1) / (k1 - k2)
        y = k1 * x + b1
    
    return (x, y)

if __name__ == "__main__":
    # 设置路径
    input_folder = "./test"  # 原始图片文件夹
    result_folder = "./pt_2_results"  # 检测结果保存文件夹
    
    # 配置两个模型
    model_configs = [
        {
            'model_path': "./models/A.pt",
            'class_name': 'A',
            'point_colors': [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        },
        {
            'model_path': "./models/CDJ.pt",  # 合并后的模型
            'class_name': 'CDJ',
            'classes': ['C', 'D', 'JB'],  # 类别列表
            'point_colors': [(0, 255, 0), (255, 0, 0)]  # 所有类别都是2个点，用相同的颜色
        }
    ]
    
    # 直接处理原始图片
    process_folder(model_configs, input_folder, result_folder)
    print("\n处理完成！") 