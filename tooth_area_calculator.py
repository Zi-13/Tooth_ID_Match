#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🦷 牙齿面积计算器 - 简化版
专门用于计算牙齿图像中黑色区域的面积

使用方法：
python tooth_area_calculator.py --image "图像路径" --reference-size 10.0
"""

import cv2
import numpy as np
import json
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ReferenceObject:
    """参考物规格定义"""
    size_mm: float = 10.0  # 真实尺寸（毫米）
    color_hsv_range: Dict = None  # HSV颜色范围
    
    def __post_init__(self):
        if self.color_hsv_range is None:
            # 扩大红色HSV范围
            self.color_hsv_range = {
                'lower': np.array([0, 50, 50]),     # 红色下界
                'upper': np.array([15, 255, 255]),  # 红色上界
                'lower2': np.array([165, 50, 50]),  # 红色下界2
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
        
    def detect_reference_object(self, image: np.ndarray) -> CalibrationResult:
        """检测图像中的参考物并计算标定参数"""
        try:
            # 转换到HSV颜色空间
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # 创建颜色掩码
            mask = self._create_color_mask(hsv)
            
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
            'bounding_rect': (x, y, w, h)
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

class ToothAreaCalculator:
    """牙齿区域面积计算器"""
    
    def __init__(self, pixel_per_mm: float):
        self.pixel_per_mm = pixel_per_mm
        
    def calculate_tooth_area(self, image: np.ndarray, save_images: bool = True) -> Dict:
        """计算牙齿（白色区域）的面积和周长"""
        try:
            # 转换为灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 使用Otsu阈值分割找到白色区域（牙齿）
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 不反转，直接使用二值图像（白色区域保持为白色）
            # binary现在包含白色的牙齿区域
            
            # 形态学操作去除噪声
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            # 闭运算填充小孔
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            # 保存中间结果图像
            if save_images:
                cv2.imwrite("tooth_binary_mask.png", cleaned)
                logger.info("牙齿二值化掩码已保存到: tooth_binary_mask.png")
            
            # 检测轮廓
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return {
                    'total_area_pixels': 0,
                    'total_area_mm2': 0,
                    'total_perimeter_pixels': 0,
                    'total_perimeter_mm': 0,
                    'contour_count': 0,
                    'largest_area_pixels': 0,
                    'largest_area_mm2': 0,
                    'largest_perimeter_pixels': 0,
                    'largest_perimeter_mm': 0,
                    'error': 'No white tooth regions found'
                }
            
            # 计算总面积、周长和最大区域
            total_area_pixels = 0
            total_perimeter_pixels = 0
            largest_area_pixels = 0
            largest_perimeter_pixels = 0
            valid_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                # 调整面积阈值，牙齿通常比较大
                if area > 500:  # 过滤小的噪声区域，牙齿区域应该较大
                    perimeter = cv2.arcLength(contour, True)
                    total_area_pixels += area
                    total_perimeter_pixels += perimeter
                    valid_contours.append(contour)
                    
                    if area > largest_area_pixels:
                        largest_area_pixels = area
                        largest_perimeter_pixels = perimeter
            
            # 转换为毫米单位
            total_area_mm2 = total_area_pixels / (self.pixel_per_mm ** 2)
            largest_area_mm2 = largest_area_pixels / (self.pixel_per_mm ** 2)
            total_perimeter_mm = total_perimeter_pixels / self.pixel_per_mm
            largest_perimeter_mm = largest_perimeter_pixels / self.pixel_per_mm
            
            # 创建结果图像
            if save_images and valid_contours:
                result_image = image.copy()
                
                # 绘制所有有效轮廓（绿色）
                cv2.drawContours(result_image, valid_contours, -1, (0, 255, 0), 2)
                
                # 高亮最大轮廓（红色）
                if valid_contours:
                    largest_contour = max(valid_contours, key=cv2.contourArea)
                    cv2.drawContours(result_image, [largest_contour], -1, (0, 0, 255), 3)
                    
                    # 在最大轮廓上标注信息
                    M = cv2.moments(largest_contour)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        
                        # 添加面积和周长标注
                        cv2.putText(result_image, f"Area: {largest_area_mm2:.1f}mm²", 
                                  (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.putText(result_image, f"Perimeter: {largest_perimeter_mm:.1f}mm", 
                                  (cx-50, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imwrite("tooth_detection_result.png", result_image)
                logger.info("牙齿检测结果已保存到: tooth_detection_result.png")
            
            result = {
                'total_area_pixels': total_area_pixels,
                'total_area_mm2': total_area_mm2,
                'total_perimeter_pixels': total_perimeter_pixels,
                'total_perimeter_mm': total_perimeter_mm,
                'contour_count': len(valid_contours),
                'largest_area_pixels': largest_area_pixels,
                'largest_area_mm2': largest_area_mm2,
                'largest_perimeter_pixels': largest_perimeter_pixels,
                'largest_perimeter_mm': largest_perimeter_mm,
                'pixel_per_mm': self.pixel_per_mm
            }
            
            return result
            
        except Exception as e:
            logger.error(f"牙齿面积计算失败: {e}")
            return {
                'total_area_pixels': 0,
                'total_area_mm2': 0,
                'total_perimeter_pixels': 0,
                'total_perimeter_mm': 0,
                'contour_count': 0,
                'largest_area_pixels': 0,
                'largest_area_mm2': 0,
                'largest_perimeter_pixels': 0,
                'largest_perimeter_mm': 0,
                'error': str(e)
            }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='牙齿面积计算器')
    parser.add_argument('--image', type=str, required=True, help='图像路径')
    parser.add_argument('--reference-size', type=float, default=10.0, help='参考物真实尺寸(mm)')
    parser.add_argument('--output', type=str, help='输出JSON结果文件路径')
    
    args = parser.parse_args()
    
    # 读取图像
    image = cv2.imread(args.image)
    if image is None:
        print(f"❌ 无法读取图像: {args.image}")
        return
    
    print(f"🔍 开始处理图像: {args.image}")
    
    # 创建参考物和检测器
    reference_obj = ReferenceObject(size_mm=args.reference_size)
    detector = ReferenceDetector(reference_obj)
    
    # 检测参考物
    calibration_result = detector.detect_reference_object(image)
    
    if calibration_result.pixel_per_mm <= 0:
        print(f"❌ 标定失败: {calibration_result.error_message}")
        return
    
    print("✅ 参考物检测成功")
    print(f"📏 缩放比例: {calibration_result.pixel_per_mm:.4f} px/mm")
    print(f"🎯 置信度: {calibration_result.confidence:.3f}")
    
    # 计算牙齿面积
    calculator = ToothAreaCalculator(calibration_result.pixel_per_mm)
    area_result = calculator.calculate_tooth_area(image, save_images=True)
    
    if 'error' in area_result:
        print(f"❌ 面积计算失败: {area_result['error']}")
        return
    
    print("\n🦷 牙齿面积和周长计算结果:")
    print("=" * 40)
    print(f"📐 总牙齿面积: {area_result['total_area_mm2']:.2f} mm²")
    print(f"� 总牙齿周长: {area_result['total_perimeter_mm']:.2f} mm")
    print(f"�📐 最大牙齿面积: {area_result['largest_area_mm2']:.2f} mm²")
    print(f"� 最大牙齿周长: {area_result['largest_perimeter_mm']:.2f} mm")
    print(f"�🔢 检测到的牙齿数量: {area_result['contour_count']}")
    print(f"📏 像素面积: {area_result['total_area_pixels']:.0f} pixels")
    print(f"📏 像素周长: {area_result['total_perimeter_pixels']:.0f} pixels")
    print(f"🔄 比例系数: {area_result['pixel_per_mm']:.4f} px/mm")
    
    # 保存结果到JSON文件
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({
                'calibration': {
                    'pixel_per_mm': calibration_result.pixel_per_mm,
                    'reference_pixel_size': calibration_result.reference_pixel_size,
                    'confidence': calibration_result.confidence
                },
                'area_results': area_result,
                'image_path': args.image
            }, f, indent=2, ensure_ascii=False)
        print(f"📄 结果已保存到: {args.output}")

if __name__ == "__main__":
    main()
