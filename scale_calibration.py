#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 牙齿图像尺寸归一化系统
基于参考物的自动尺度标定和特征标准化

功能：
1. 检测标准参考物（红色正方体）
2. 计算像素/毫米比例
3. 标准化几何特征（面积、周长）
4. 标准化傅里叶描述符
5. 可视化标定结果

作者：AI Assistant
创建时间：2025-07-23
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import argparse

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ReferenceObject:
    """参考物规格定义"""
    size_mm: float = 10.0  # 真实尺寸（毫米）
    color_hsv_range: Dict = None  # HSV颜色范围
    shape: str = "square"  # 形状类型
    
    def __post_init__(self):
        if self.color_hsv_range is None:
            # 扩大红色HSV范围以适应更多红色变化
            self.color_hsv_range = {
                'lower': np.array([0, 50, 50]),     # 红色下界（更宽松）
                'upper': np.array([15, 255, 255]),  # 红色上界（更宽松）
                'lower2': np.array([165, 50, 50]),  # 红色下界2（跨越0度，更宽松）
                'upper2': np.array([180, 255, 255]) # 红色上界2
            }

@dataclass
class CalibrationResult:
    """标定结果数据类"""
    pixel_per_mm: float  # 像素/毫米比例
    reference_pixel_size: float  # 参考物像素尺寸
    reference_position: Tuple[int, int, int, int]  # 参考物位置 (x, y, w, h)
    confidence: float  # 置信度 (0-1)
    error_message: str = ""  # 错误信息

class ReferenceDetector:
    """参考物检测器"""
    
    def __init__(self, reference_obj: ReferenceObject):
        self.reference_obj = reference_obj
        
    def detect_reference_object(self, image: np.ndarray) -> Optional[CalibrationResult]:
        """
        检测图像中的参考物并计算标定参数
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            CalibrationResult: 标定结果，失败返回None
        """
        try:
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 创建颜色掩码
            mask = self._create_color_mask(hsv)
            
            # 调试：保存掩码图像以便检查
            debug_mask_path = "debug_red_mask.png"
            cv2.imwrite(debug_mask_path, mask)
            logger.info(f"红色掩码已保存到: {debug_mask_path}")
            
            # 形态学操作清理掩码
            mask = self._clean_mask(mask)
            
            # 检测轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return CalibrationResult(0, 0, (0, 0, 0, 0), 0, "未检测到参考物颜色")
            
            # 筛选和验证参考物
            best_contour = self._find_best_reference_contour(contours)
            
            if best_contour is None:
                return CalibrationResult(0, 0, (0, 0, 0, 0), 0, "未找到符合条件的参考物")
            
            # 计算标定参数
            return self._calculate_calibration(best_contour)
            
        except Exception as e:
            logger.error(f"参考物检测失败: {e}")
            return CalibrationResult(0, 0, (0, 0, 0, 0), 0, f"检测异常: {str(e)}")
    
    def _create_color_mask(self, hsv: np.ndarray) -> np.ndarray:
        """创建颜色掩码"""
        color_range = self.reference_obj.color_hsv_range
        
        # 红色可能跨越HSV的0度，需要处理两个范围
        mask1 = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
        
        # 合并两个掩码
        mask = cv2.bitwise_or(mask1, mask2)
        
        return mask
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """清理掩码噪声"""
        # 开运算去除小噪声
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # 闭运算填充小孔
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask
    
    def _find_best_reference_contour(self, contours: List) -> Optional[np.ndarray]:
        """找到最佳的参考物轮廓"""
        candidates = []
        
        for contour in contours:
            # 基本面积筛选
            area = cv2.contourArea(contour)
            if area < 100:  # 太小的区域忽略
                continue
            
            # 计算轮廓特征
            features = self._analyze_contour_shape(contour)
            
            # 评估是否符合参考物要求
            score = self._evaluate_reference_candidate(features)
            
            if score > 0.5:  # 置信度阈值
                candidates.append((contour, score, features))
        
        if not candidates:
            return None
        
        # 返回得分最高的候选
        best_contour, best_score, best_features = max(candidates, key=lambda x: x[1])
        logger.info(f"最佳参考物候选得分: {best_score:.3f}")
        
        return best_contour
    
    def _analyze_contour_shape(self, contour: np.ndarray) -> Dict:
        """分析轮廓形状特征"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 外接矩形
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # 最小外接矩形
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # 计算矩形度
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # 计算圆形度
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
        # 计算凸包特征
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'rectangularity': rectangularity,
            'circularity': circularity,
            'solidity': solidity,
            'bounding_rect': (x, y, w, h),
            'min_area_rect': rect,
            'contour_points': len(contour)
        }
    
    def _evaluate_reference_candidate(self, features: Dict) -> float:
        """评估参考物候选的质量"""
        score = 0.0
        
        # 长宽比评估（正方形应该接近1）
        aspect_ratio = features['aspect_ratio']
        if 0.8 <= aspect_ratio <= 1.25:  # 允许一定的检测误差
            score += 0.3
        
        # 矩形度评估（应该是矩形）
        rectangularity = features['rectangularity']
        if rectangularity > 0.7:
            score += 0.3
        
        # 实心度评估（应该是实心的）
        solidity = features['solidity']
        if solidity > 0.8:
            score += 0.2
        
        # 面积合理性评估（不能太大也不能太小）
        area = features['area']
        if 100 <= area <= 10000:  # 合理的像素面积范围
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_calibration(self, contour: np.ndarray) -> CalibrationResult:
        """计算标定参数"""
        # 计算参考物的像素尺寸
        x, y, w, h = cv2.boundingRect(contour)
        pixel_size = (w + h) / 2  # 使用宽高平均值作为像素尺寸
        
        # 计算像素/毫米比例
        pixel_per_mm = pixel_size / self.reference_obj.size_mm
        
        # 计算置信度（基于正方形程度）
        aspect_ratio = w / h if h > 0 else 0
        confidence = 1.0 - abs(1.0 - aspect_ratio)  # 越接近1（正方形）置信度越高
        confidence = max(0.0, min(1.0, confidence))
        
        logger.info(f"参考物检测成功:")
        logger.info(f"  像素尺寸: {pixel_size:.2f} pixels")
        logger.info(f"  真实尺寸: {self.reference_obj.size_mm} mm")
        logger.info(f"  比例系数: {pixel_per_mm:.4f} pixels/mm")
        logger.info(f"  置信度: {confidence:.3f}")
        
        return CalibrationResult(
            pixel_per_mm=pixel_per_mm,
            reference_pixel_size=pixel_size,
            reference_position=(x, y, w, h),
            confidence=confidence
        )

class FeatureNormalizer:
    """特征标准化器"""
    
    def __init__(self, standard_pixel_per_mm: float = 10.0):
        """
        初始化标准化器
        
        Args:
            standard_pixel_per_mm: 标准比例（像素/毫米），所有特征将标准化到此比例
        """
        self.standard_pixel_per_mm = standard_pixel_per_mm
    
    def normalize_features(self, features: Dict, source_pixel_per_mm: float) -> Dict:
        """
        标准化特征到统一尺度
        
        Args:
            features: 原始特征字典
            source_pixel_per_mm: 源图像的像素/毫米比例
            
        Returns:
            标准化后的特征字典
        """
        if source_pixel_per_mm <= 0:
            logger.warning("无效的像素比例，返回原始特征")
            return features.copy()
        
        # 计算缩放因子
        scale_factor = source_pixel_per_mm / self.standard_pixel_per_mm
        
        normalized_features = features.copy()
        
        # 标准化面积特征（缩放因子的平方）
        if 'area' in features:
            normalized_features['area'] = features['area'] / (scale_factor ** 2)
        
        # 标准化周长特征（缩放因子的一次方）
        if 'perimeter' in features:
            normalized_features['perimeter'] = features['perimeter'] / scale_factor
        
        # 标准化归一化面积和周长（如果存在）
        if 'area_norm' in features:
            normalized_features['area_norm'] = features['area_norm'] / (scale_factor ** 2)
        if 'perimeter_norm' in features:
            normalized_features['perimeter_norm'] = features['perimeter_norm'] / scale_factor
        
        # 标准化傅里叶描述符（0阶系数需要缩放）
        if 'fourier_descriptors' in features:
            fourier_desc = np.array(features['fourier_descriptors']).copy()
            if len(fourier_desc) >= 2:
                # 0阶傅里叶系数对应轮廓的平移，需要缩放
                fourier_desc[0] /= scale_factor  # X方向0阶系数
                fourier_desc[11] /= scale_factor if len(fourier_desc) > 11 else 1  # Y方向0阶系数
            normalized_features['fourier_descriptors'] = fourier_desc.tolist()
        
        # 保留不变特征（长宽比、圆形度、实心度等）
        invariant_features = ['aspect_ratio', 'circularity', 'solidity', 'corner_count', 'hu_moments']
        for feat in invariant_features:
            if feat in features:
                normalized_features[feat] = features[feat]
        
        # 记录标准化信息
        normalized_features['scale_factor'] = scale_factor
        normalized_features['source_pixel_per_mm'] = source_pixel_per_mm
        normalized_features['standard_pixel_per_mm'] = self.standard_pixel_per_mm
        
        logger.info(f"特征标准化完成，缩放因子: {scale_factor:.4f}")
        
        return normalized_features

class ToothAreaCalculator:
    """牙齿区域面积计算器"""
    
    def __init__(self, pixel_per_mm: float):
        """
        初始化面积计算器
        
        Args:
            pixel_per_mm: 像素/毫米比例
        """
        self.pixel_per_mm = pixel_per_mm
        
    def calculate_tooth_area(self, image: np.ndarray, visualization: bool = True) -> Dict:
        """
        计算牙齿（黑色区域）的面积
        
        Args:
            image: 输入图像
            visualization: 是否显示可视化结果
            
        Returns:
            包含面积信息的字典
        """
        try:
            # 转换为灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 使用Otsu阈值分割找到黑色区域
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 反转二值图像，使黑色区域变为白色
            binary_inv = cv2.bitwise_not(binary)
            
            # 形态学操作去除噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
            
            # 闭运算填充小孔
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # 检测轮廓
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {
                    'total_area_pixels': 0,
                    'total_area_mm2': 0,
                    'contour_count': 0,
                    'largest_area_pixels': 0,
                    'largest_area_mm2': 0,
                    'error': 'No dark regions found'
                }
            
            # 计算总面积和最大区域面积
            total_area_pixels = 0
            largest_area_pixels = 0
            largest_contour = None
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:  # 过滤小的噪声区域
                    total_area_pixels += area
                    if area > largest_area_pixels:
                        largest_area_pixels = area
                        largest_contour = contour
            
            # 转换为毫米单位
            total_area_mm2 = total_area_pixels / (self.pixel_per_mm ** 2)
            largest_area_mm2 = largest_area_pixels / (self.pixel_per_mm ** 2)
            
            result = {
                'total_area_pixels': total_area_pixels,
                'total_area_mm2': total_area_mm2,
                'contour_count': len([c for c in contours if cv2.contourArea(c) > 100]),
                'largest_area_pixels': largest_area_pixels,
                'largest_area_mm2': largest_area_mm2,
                'pixel_per_mm': self.pixel_per_mm
            }
            
            if visualization:
                self._visualize_tooth_detection(image, cleaned, contours, result)
            
            return result
            
        except Exception as e:
            logger.error(f"牙齿面积计算失败: {e}")
            return {
                'total_area_pixels': 0,
                'total_area_mm2': 0,
                'contour_count': 0,
                'largest_area_pixels': 0,
                'largest_area_mm2': 0,
                'error': str(e)
            }
    
    def _visualize_tooth_detection(self, original: np.ndarray, binary: np.ndarray, 
                                 contours: List, result: Dict) -> None:
        """可视化牙齿检测结果"""
        # 创建结果图像
        result_image = original.copy()
        
        # 绘制所有有效轮廓
        valid_contours = [c for c in contours if cv2.contourArea(c) > 100]
        cv2.drawContours(result_image, valid_contours, -1, (0, 255, 0), 2)
        
        # 绘制最大轮廓
        if valid_contours:
            largest_contour = max(valid_contours, key=cv2.contourArea)
            cv2.drawContours(result_image, [largest_contour], -1, (0, 0, 255), 3)
        
        # 显示结果
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Original Image", fontsize=12)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title("Binary Mask (Dark Regions)", fontsize=12)
        plt.imshow(binary, cmap='gray')
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title("Detection Result", fontsize=12)
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # 添加结果信息
        info_text = f"""Area Calculation Results:
• Total Dark Area: {result['total_area_mm2']:.2f} mm²
• Largest Region: {result['largest_area_mm2']:.2f} mm²
• Number of Regions: {result['contour_count']}
• Scale: {result['pixel_per_mm']:.3f} px/mm
        """
        
        plt.figtext(0.02, 0.02, info_text, fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"))
        
        plt.tight_layout()
        plt.show()

class ScaleCalibrationSystem:
    """尺度标定系统主类"""
    
    def __init__(self, reference_obj: ReferenceObject = None, 
                 standard_pixel_per_mm: float = 10.0):
        """
        初始化标定系统
        
        Args:
            reference_obj: 参考物规格
            standard_pixel_per_mm: 标准像素/毫米比例
        """
        self.reference_obj = reference_obj or ReferenceObject()
        self.detector = ReferenceDetector(self.reference_obj)
        self.normalizer = FeatureNormalizer(standard_pixel_per_mm)
        self.tooth_calculator = None  # 将在标定后初始化
        
    def calibrate_and_calculate_area(self, image_path: str) -> Tuple[Optional[CalibrationResult], Dict]:
        """
        标定图像并计算牙齿面积
        
        Args:
            image_path: 图像路径
            
        Returns:
            (标定结果, 面积计算结果)
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return None, {}
        
        logger.info(f"开始处理图像: {image_path}")
        
        # 标定尺度
        calibration_result = self.detector.detect_reference_object(image)
        
        if calibration_result.pixel_per_mm <= 0:
            logger.error(f"标定失败: {calibration_result.error_message}")
            return calibration_result, {}
        
        logger.info("✅ 图像标定成功")
        
        # 初始化牙齿面积计算器
        self.tooth_calculator = ToothAreaCalculator(calibration_result.pixel_per_mm)
        
        # 计算牙齿面积
        area_result = self.tooth_calculator.calculate_tooth_area(image, visualization=True)
        
        logger.info("✅ 面积计算完成")
        logger.info(f"   总黑色面积: {area_result.get('total_area_mm2', 0):.2f} mm²")
        logger.info(f"   最大区域面积: {area_result.get('largest_area_mm2', 0):.2f} mm²")
        
        return calibration_result, area_result
        
    def calibrate_image(self, image_path: str) -> Tuple[Optional[CalibrationResult], np.ndarray]:
        """
        标定图像尺度
        
        Args:
            image_path: 图像路径
            
        Returns:
            (标定结果, 图像数据)
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return None, None
        
        logger.info(f"开始标定图像: {image_path}")
        
        # 检测参考物
        calibration_result = self.detector.detect_reference_object(image)
        
        if calibration_result.pixel_per_mm <= 0:
            logger.error(f"标定失败: {calibration_result.error_message}")
            return calibration_result, image
        
        logger.info("✅ 图像标定成功")
        return calibration_result, image
    
    def normalize_image_features(self, features: Dict, calibration_result: CalibrationResult) -> Dict:
        """
        标准化图像特征
        
        Args:
            features: 待标准化的特征
            calibration_result: 标定结果
            
        Returns:
            标准化后的特征
        """
        return self.normalizer.normalize_features(features, calibration_result.pixel_per_mm)
    
    def visualize_calibration(self, image: np.ndarray, calibration_result: CalibrationResult, 
                            save_path: str = None) -> None:
        """
        可视化标定结果
        
        Args:
            image: 原始图像
            calibration_result: 标定结果
            save_path: 保存路径（可选）
        """
        if calibration_result.pixel_per_mm <= 0:
            logger.warning("标定失败，无法可视化")
            return
        
        # 创建可视化图像
        vis_image = image.copy()
        x, y, w, h = calibration_result.reference_position
        
        # 绘制参考物边界框
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 添加标注信息
        info_text = [
            f"Reference: {self.reference_obj.size_mm}mm",
            f"Pixels: {calibration_result.reference_pixel_size:.1f}px",
            f"Scale: {calibration_result.pixel_per_mm:.3f}px/mm",
            f"Confidence: {calibration_result.confidence:.3f}"
        ]
        
        # 在图像上绘制文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        for i, text in enumerate(info_text):
            y_offset = 30 + i * 25
            cv2.putText(vis_image, text, (10, y_offset), font, font_scale, (0, 255, 0), thickness)
        
        # 使用matplotlib显示
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.title("Original Image", fontsize=14)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.title("Calibration Result", fontsize=14)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        # 添加详细信息文本
        info_str = f"""Calibration Results:
• Reference Size: {self.reference_obj.size_mm} mm
• Pixel Size: {calibration_result.reference_pixel_size:.2f} px
• Scale Ratio: {calibration_result.pixel_per_mm:.4f} px/mm
• Confidence: {calibration_result.confidence:.3f}
        """
        plt.figtext(0.02, 0.02, info_str, fontsize=11, 
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"可视化结果已保存: {save_path}")
        
        plt.show()

def test_calibration_system():
    """测试标定系统"""
    import os
    
    print("🎯 尺度标定系统测试")
    print("=" * 50)
    
    # 创建标定系统
    system = ScaleCalibrationSystem()
    
    # 查找测试图像
    test_images = []
    image_dir = Path("images")
    if image_dir.exists():
        test_images = list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    
    if not test_images:
        print("❌ 未找到测试图像")
        print("💡 请在 images/ 目录中放入包含红色正方体参考物的图像")
        return
    
    print(f"📸 找到 {len(test_images)} 个测试图像")
    
    for image_path in test_images[:3]:  # 测试前3个图像
        print(f"\n🔍 测试图像: {image_path.name}")
        
        # 标定图像
        calibration_result, image = system.calibrate_image(str(image_path))
        
        if calibration_result and calibration_result.pixel_per_mm > 0:
            print(f"✅ 标定成功")
            print(f"   像素比例: {calibration_result.pixel_per_mm:.4f} px/mm")
            print(f"   置信度: {calibration_result.confidence:.3f}")
            
            # 可视化结果
            system.visualize_calibration(image, calibration_result)
            
            # 测试特征标准化
            sample_features = {
                'area': 1000.0,
                'perimeter': 200.0,
                'aspect_ratio': 1.2,
                'circularity': 0.8,
                'fourier_descriptors': [10.0, 5.0, 3.0] + [0.0] * 19
            }
            
            normalized_features = system.normalize_image_features(sample_features, calibration_result)
            
            print(f"   特征标准化测试:")
            print(f"     原始面积: {sample_features['area']:.1f}")
            print(f"     标准面积: {normalized_features['area']:.1f}")
            print(f"     缩放因子: {normalized_features['scale_factor']:.4f}")
            
        else:
            print(f"❌ 标定失败: {calibration_result.error_message if calibration_result else '未知错误'}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='牙齿图像尺度标定系统')
    parser.add_argument('--test', action='store_true', help='运行测试模式')
    parser.add_argument('--image', type=str, help='标定指定图像')
    parser.add_argument('--reference-size', type=float, default=10.0, help='参考物真实尺寸(mm)')
    parser.add_argument('--output', type=str, help='输出可视化结果路径')
    parser.add_argument('--calculate-area', action='store_true', help='计算黑色区域面积')
    
    args = parser.parse_args()
    
    if args.test:
        test_calibration_system()
    elif args.image:
        # 标定指定图像
        reference_obj = ReferenceObject(size_mm=args.reference_size)
        system = ScaleCalibrationSystem(reference_obj)
        
        if args.calculate_area:
            # 标定并计算面积
            calibration_result, area_result = system.calibrate_and_calculate_area(args.image)
            
            if calibration_result and calibration_result.pixel_per_mm > 0:
                print("✅ 标定和面积计算成功")
                print(f"📏 缩放比例: {calibration_result.pixel_per_mm:.4f} px/mm")
                print(f"📐 总黑色面积: {area_result.get('total_area_mm2', 0):.2f} mm²")
                print(f"📐 最大区域面积: {area_result.get('largest_area_mm2', 0):.2f} mm²")
                print(f"🔢 检测到的区域数量: {area_result.get('contour_count', 0)}")
            else:
                print("❌ 标定失败")
        else:
            # 仅标定
            calibration_result, image = system.calibrate_image(args.image)
            
            if calibration_result and calibration_result.pixel_per_mm > 0:
                print("✅ 标定成功")
                system.visualize_calibration(image, calibration_result, args.output)
            else:
                print("❌ 标定失败")
    else:
        test_calibration_system()

if __name__ == "__main__":
    main()
