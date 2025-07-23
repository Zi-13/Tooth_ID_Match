#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🦷 牙齿面积分析交互界面
基于Tkinter的GUI应用程序

功能：
1. 选择图像文件
2. 实时显示检测结果
3. 交互式参数调整
4. 结果保存和导出

作者：AI Assistant
创建时间：2025-07-23
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import json
import threading
from pathlib import Path
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ReferenceObject:
    """参考物规格定义"""
    size_mm: float = 10.0
    color_hsv_range: Dict = None
    
    def __post_init__(self):
        if self.color_hsv_range is None:
            self.color_hsv_range = {
                'lower': np.array([0, 50, 50]),
                'upper': np.array([15, 255, 255]),
                'lower2': np.array([165, 50, 50]),
                'upper2': np.array([180, 255, 255])
            }

@dataclass
class CalibrationResult:
    """标定结果数据类"""
    pixel_per_mm: float
    reference_pixel_size: float
    reference_position: Tuple[int, int, int, int]
    confidence: float
    error_message: str = ""

class ReferenceDetector:
    """参考物检测器"""
    
    def __init__(self, reference_obj: ReferenceObject):
        self.reference_obj = reference_obj
        
    def detect_reference_object(self, image: np.ndarray) -> CalibrationResult:
        """检测图像中的参考物并计算标定参数"""
        try:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            mask = self._create_color_mask(hsv)
            mask = self._clean_mask(mask)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return CalibrationResult(0, 0, (0, 0, 0, 0), 0, "未检测到参考物颜色")
            
            best_contour = self._find_best_reference_contour(contours)
            
            if best_contour is None:
                return CalibrationResult(0, 0, (0, 0, 0, 0), 0, "未找到符合条件的参考物")
            
            return self._calculate_calibration(best_contour)
            
        except Exception as e:
            logger.error(f"参考物检测失败: {e}")
            return CalibrationResult(0, 0, (0, 0, 0, 0), 0, f"检测异常: {str(e)}")
    
    def _create_color_mask(self, hsv: np.ndarray) -> np.ndarray:
        """创建颜色掩码"""
        color_range = self.reference_obj.color_hsv_range
        mask1 = cv2.inRange(hsv, color_range['lower'], color_range['upper'])
        mask2 = cv2.inRange(hsv, color_range['lower2'], color_range['upper2'])
        return cv2.bitwise_or(mask1, mask2)
    
    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """清理掩码噪声"""
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask
    
    def _find_best_reference_contour(self, contours: List) -> Optional[np.ndarray]:
        """找到最佳的参考物轮廓"""
        candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:
                continue
            
            features = self._analyze_contour_shape(contour)
            score = self._evaluate_reference_candidate(features)
            
            if score > 0.5:
                candidates.append((contour, score, features))
        
        if not candidates:
            return None
        
        best_contour, best_score, best_features = max(candidates, key=lambda x: x[1])
        return best_contour
    
    def _analyze_contour_shape(self, contour: np.ndarray) -> Dict:
        """分析轮廓形状特征"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
        
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
        
        aspect_ratio = features['aspect_ratio']
        if 0.8 <= aspect_ratio <= 1.25:
            score += 0.3
        
        rectangularity = features['rectangularity']
        if rectangularity > 0.7:
            score += 0.3
        
        solidity = features['solidity']
        if solidity > 0.8:
            score += 0.2
        
        area = features['area']
        if 100 <= area <= 10000:
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_calibration(self, contour: np.ndarray) -> CalibrationResult:
        """计算标定参数"""
        x, y, w, h = cv2.boundingRect(contour)
        pixel_size = (w + h) / 2
        pixel_per_mm = pixel_size / self.reference_obj.size_mm
        
        aspect_ratio = w / h if h > 0 else 0
        confidence = 1.0 - abs(1.0 - aspect_ratio)
        confidence = max(0.0, min(1.0, confidence))
        
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
        
    def calculate_tooth_area(self, image: np.ndarray) -> Tuple[Dict, np.ndarray, np.ndarray]:
        """计算牙齿（白色区域）面积并返回处理过程图像"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 不反转，直接使用二值图像（白色牙齿区域）
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
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
                }, cleaned, image.copy()
            
            total_area_pixels = 0
            total_perimeter_pixels = 0
            largest_area_pixels = 0
            largest_perimeter_pixels = 0
            valid_contours = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:  # 牙齿应该较大
                    perimeter = cv2.arcLength(contour, True)
                    total_area_pixels += area
                    total_perimeter_pixels += perimeter
                    valid_contours.append(contour)
                    
                    if area > largest_area_pixels:
                        largest_area_pixels = area
                        largest_perimeter_pixels = perimeter
            
            total_area_mm2 = total_area_pixels / (self.pixel_per_mm ** 2)
            largest_area_mm2 = largest_area_pixels / (self.pixel_per_mm ** 2)
            total_perimeter_mm = total_perimeter_pixels / self.pixel_per_mm
            largest_perimeter_mm = largest_perimeter_pixels / self.pixel_per_mm
            
            # 创建结果图像
            result_image = image.copy()
            cv2.drawContours(result_image, valid_contours, -1, (0, 255, 0), 2)
            
            if valid_contours:
                largest_contour = max(valid_contours, key=cv2.contourArea)
                cv2.drawContours(result_image, [largest_contour], -1, (0, 0, 255), 3)
                
                # 添加标注
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(result_image, f"Area: {largest_area_mm2:.1f}mm²", 
                              (cx-50, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(result_image, f"Perimeter: {largest_perimeter_mm:.1f}mm", 
                              (cx-50, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
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
            
            return result, cleaned, result_image
            
        except Exception as e:
            logger.error(f"牙齿面积计算失败: {e}")
            return {
                'total_area_pixels': 0,
                'total_area_mm2': 0,
                'contour_count': 0,
                'largest_area_pixels': 0,
                'largest_area_mm2': 0,
                'error': str(e)
            }, image, image

class ToothAnalysisGUI:
    """牙齿分析GUI主类"""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("🦷 牙齿面积分析系统")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # 数据存储
        self.current_image = None
        self.original_image = None
        self.calibration_result = None
        self.area_result = None
        self.reference_size = tk.DoubleVar(value=10.0)
        
        self.setup_ui()
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(1, weight=1)
        
        # 控制面板
        self.setup_control_panel(main_frame)
        
        # 图像显示区域
        self.setup_image_panel(main_frame)
        
        # 结果显示区域
        self.setup_result_panel(main_frame)
        
    def setup_control_panel(self, parent):
        """设置控制面板"""
        control_frame = ttk.LabelFrame(parent, text="控制面板", padding="10")
        control_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 文件选择
        ttk.Button(control_frame, text="📁 选择图像", command=self.select_image).grid(row=0, column=0, padx=(0, 10))
        
        # 参考物尺寸设置
        ttk.Label(control_frame, text="参考物尺寸(mm):").grid(row=0, column=1, padx=(0, 5))
        ttk.Entry(control_frame, textvariable=self.reference_size, width=10).grid(row=0, column=2, padx=(0, 10))
        
        # 分析按钮
        ttk.Button(control_frame, text="🔍 开始分析", command=self.analyze_image, style="Accent.TButton").grid(row=0, column=3, padx=(0, 10))
        
        # 保存结果按钮
        ttk.Button(control_frame, text="💾 保存结果", command=self.save_results).grid(row=0, column=4, padx=(0, 10))
        
        # 状态标签
        self.status_label = ttk.Label(control_frame, text="请选择图像文件开始分析", foreground="blue")
        self.status_label.grid(row=0, column=5, padx=(10, 0))
        
    def setup_image_panel(self, parent):
        """设置图像显示面板"""
        image_frame = ttk.LabelFrame(parent, text="图像显示", padding="10")
        image_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # 创建Notebook用于多标签页显示
        self.image_notebook = ttk.Notebook(image_frame)
        self.image_notebook.pack(fill=tk.BOTH, expand=True)
        
        # 原始图像标签页
        self.original_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.original_tab, text="原始图像")
        
        self.original_canvas = tk.Canvas(self.original_tab, bg='white', width=400, height=300)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 二值化标签页
        self.binary_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.binary_tab, text="二值化结果")
        
        self.binary_canvas = tk.Canvas(self.binary_tab, bg='white', width=400, height=300)
        self.binary_canvas.pack(fill=tk.BOTH, expand=True)
        
        # 检测结果标签页
        self.result_tab = ttk.Frame(self.image_notebook)
        self.image_notebook.add(self.result_tab, text="检测结果")
        
        self.result_canvas = tk.Canvas(self.result_tab, bg='white', width=400, height=300)
        self.result_canvas.pack(fill=tk.BOTH, expand=True)
        
    def setup_result_panel(self, parent):
        """设置结果显示面板"""
        result_frame = ttk.LabelFrame(parent, text="分析结果", padding="10")
        result_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标定结果
        calib_frame = ttk.LabelFrame(result_frame, text="标定结果", padding="10")
        calib_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.calib_text = tk.Text(calib_frame, height=8, width=50, font=("Consolas", 10))
        calib_scrollbar = ttk.Scrollbar(calib_frame, orient=tk.VERTICAL, command=self.calib_text.yview)
        self.calib_text.configure(yscrollcommand=calib_scrollbar.set)
        
        self.calib_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        calib_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 面积计算结果
        area_frame = ttk.LabelFrame(result_frame, text="面积计算结果", padding="10")
        area_frame.pack(fill=tk.BOTH, expand=True)
        
        self.area_text = tk.Text(area_frame, height=12, width=50, font=("Consolas", 10))
        area_scrollbar = ttk.Scrollbar(area_frame, orient=tk.VERTICAL, command=self.area_text.yview)
        self.area_text.configure(yscrollcommand=area_scrollbar.set)
        
        self.area_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        area_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
    def select_image(self):
        """选择图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[
                ("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG文件", "*.jpg *.jpeg"),
                ("PNG文件", "*.png"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    messagebox.showerror("错误", "无法读取图像文件")
                    return
                
                self.current_image = self.original_image.copy()
                self.display_image(self.original_canvas, self.current_image)
                self.status_label.config(text=f"已加载: {Path(file_path).name}", foreground="green")
                
                # 清空结果显示
                self.clear_results()
                
            except Exception as e:
                messagebox.showerror("错误", f"加载图像失败: {str(e)}")
    
    def analyze_image(self):
        """分析图像"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先选择图像文件")
            return
        
        # 在后台线程中执行分析
        self.status_label.config(text="正在分析...", foreground="orange")
        self.root.config(cursor="wait")
        
        thread = threading.Thread(target=self._analyze_worker)
        thread.daemon = True
        thread.start()
    
    def _analyze_worker(self):
        """后台分析工作线程"""
        try:
            # 检测参考物
            reference_obj = ReferenceObject(size_mm=self.reference_size.get())
            detector = ReferenceDetector(reference_obj)
            self.calibration_result = detector.detect_reference_object(self.original_image)
            
            if self.calibration_result.pixel_per_mm <= 0:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"标定失败: {self.calibration_result.error_message}", foreground="red"))
                self.root.after(0, lambda: self.root.config(cursor=""))
                return
            
            # 计算牙齿面积
            calculator = ToothAreaCalculator(self.calibration_result.pixel_per_mm)
            self.area_result, binary_image, result_image = calculator.calculate_tooth_area(self.original_image)
            
            # 在主线程中更新UI
            self.root.after(0, lambda: self.update_results(binary_image, result_image))
            
        except Exception as e:
            logger.error(f"分析失败: {e}")
            self.root.after(0, lambda: self.status_label.config(text=f"分析失败: {str(e)}", foreground="red"))
            self.root.after(0, lambda: self.root.config(cursor=""))
    
    def update_results(self, binary_image, result_image):
        """更新结果显示"""
        try:
            # 显示图像
            self.display_image(self.binary_canvas, binary_image, is_gray=True)
            self.display_image(self.result_canvas, result_image)
            
            # 显示标定结果
            self.display_calibration_results()
            
            # 显示面积结果
            self.display_area_results()
            
            self.status_label.config(text="分析完成", foreground="green")
            self.root.config(cursor="")
            
        except Exception as e:
            logger.error(f"更新结果失败: {e}")
            self.status_label.config(text=f"更新结果失败: {str(e)}", foreground="red")
            self.root.config(cursor="")
    
    def display_image(self, canvas, image, is_gray=False):
        """在Canvas上显示图像"""
        try:
            if is_gray and len(image.shape) == 2:
                # 灰度图像
                pil_image = Image.fromarray(image)
            else:
                # 彩色图像，从BGR转换为RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
            
            # 调整图像大小以适应Canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                # Canvas尚未初始化，使用默认大小
                canvas_width, canvas_height = 400, 300
            
            img_width, img_height = pil_image.size
            
            # 计算缩放比例
            scale_w = canvas_width / img_width
            scale_h = canvas_height / img_height
            scale = min(scale_w, scale_h) * 0.9  # 留一些边距
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可用的格式
            photo = ImageTk.PhotoImage(pil_image)
            
            # 清空Canvas并显示图像
            canvas.delete("all")
            x = (canvas_width - new_width) // 2
            y = (canvas_height - new_height) // 2
            canvas.create_image(x, y, anchor=tk.NW, image=photo)
            
            # 保存引用以防止垃圾回收
            canvas.image = photo
            
        except Exception as e:
            logger.error(f"显示图像失败: {e}")
    
    def display_calibration_results(self):
        """显示标定结果"""
        self.calib_text.delete(1.0, tk.END)
        
        if self.calibration_result:
            result_text = f"""🎯 参考物检测结果
{'='*40}
✅ 检测状态: 成功
📏 参考物尺寸: {self.reference_size.get():.1f} mm
📐 像素尺寸: {self.calibration_result.reference_pixel_size:.2f} pixels
🔄 比例系数: {self.calibration_result.pixel_per_mm:.4f} px/mm
🎯 置信度: {self.calibration_result.confidence:.3f} ({self.calibration_result.confidence*100:.1f}%)

📍 参考物位置:
   X: {self.calibration_result.reference_position[0]}
   Y: {self.calibration_result.reference_position[1]} 
   宽度: {self.calibration_result.reference_position[2]}
   高度: {self.calibration_result.reference_position[3]}
"""
        else:
            result_text = "❌ 暂无标定结果"
        
        self.calib_text.insert(tk.END, result_text)
    
    def display_area_results(self):
        """显示面积计算结果"""
        self.area_text.delete(1.0, tk.END)
        
        if self.area_result and 'error' not in self.area_result:
            result_text = f"""🦷 牙齿面积和周长分析结果
{'='*40}
📐 总牙齿面积: {self.area_result['total_area_mm2']:.2f} mm²
📏 总牙齿周长: {self.area_result.get('total_perimeter_mm', 0):.2f} mm
📐 最大牙齿面积: {self.area_result['largest_area_mm2']:.2f} mm²
📏 最大牙齿周长: {self.area_result.get('largest_perimeter_mm', 0):.2f} mm
🔢 检测到的牙齿数量: {self.area_result['contour_count']}
📏 像素面积: {self.area_result['total_area_pixels']:.0f} pixels
� 像素周长: {self.area_result.get('total_perimeter_pixels', 0):.0f} pixels
�🔄 比例系数: {self.area_result['pixel_per_mm']:.4f} px/mm

📊 详细信息:
   最大牙齿占比: {(self.area_result['largest_area_mm2']/self.area_result['total_area_mm2']*100) if self.area_result['total_area_mm2'] > 0 else 0:.1f}%
   平均牙齿面积: {(self.area_result['total_area_mm2']/self.area_result['contour_count']) if self.area_result['contour_count'] > 0 else 0:.2f} mm²
   平均牙齿周长: {(self.area_result.get('total_perimeter_mm', 0)/self.area_result['contour_count']) if self.area_result['contour_count'] > 0 else 0:.2f} mm
   
💡 说明:
   - 绿色轮廓: 所有检测到的牙齿区域
   - 红色轮廓: 最大的牙齿
   - 面积单位: 平方毫米(mm²)
   - 周长单位: 毫米(mm)
"""
        elif self.area_result and 'error' in self.area_result:
            result_text = f"❌ 面积计算失败: {self.area_result['error']}"
        else:
            result_text = "❌ 暂无面积计算结果"
        
        self.area_text.insert(tk.END, result_text)
    
    def clear_results(self):
        """清空结果显示"""
        self.calib_text.delete(1.0, tk.END)
        self.area_text.delete(1.0, tk.END)
        
        # 清空图像显示
        for canvas in [self.binary_canvas, self.result_canvas]:
            canvas.delete("all")
            if hasattr(canvas, 'image'):
                del canvas.image
    
    def save_results(self):
        """保存分析结果"""
        if not self.calibration_result or not self.area_result:
            messagebox.showwarning("警告", "请先进行图像分析")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存分析结果",
            defaultextension=".json",
            filetypes=[
                ("JSON文件", "*.json"),
                ("文本文件", "*.txt"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            try:
                # 准备保存的数据
                save_data = {
                    "calibration_results": {
                        "pixel_per_mm": self.calibration_result.pixel_per_mm,
                        "reference_pixel_size": self.calibration_result.reference_pixel_size,
                        "reference_position": self.calibration_result.reference_position,
                        "confidence": self.calibration_result.confidence,
                        "reference_size_mm": self.reference_size.get()
                    },
                    "area_results": self.area_result,
                    "analysis_timestamp": str(Path.cwd() / "analysis_" + str(hash(str(self.area_result))))
                }
                
                if file_path.endswith('.json'):
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, indent=2, ensure_ascii=False)
                else:
                    # 保存为文本格式
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write("🦷 牙齿面积分析结果报告\n")
                        f.write("="*50 + "\n\n")
                        
                        f.write("📏 标定结果:\n")
                        f.write(f"  参考物尺寸: {self.reference_size.get():.1f} mm\n")
                        f.write(f"  比例系数: {self.calibration_result.pixel_per_mm:.4f} px/mm\n")
                        f.write(f"  置信度: {self.calibration_result.confidence:.3f}\n\n")
                        
                        f.write("📐 面积结果:\n")
                        f.write(f"  总面积: {self.area_result['total_area_mm2']:.2f} mm²\n")
                        f.write(f"  最大区域: {self.area_result['largest_area_mm2']:.2f} mm²\n")
                        f.write(f"  区域数量: {self.area_result['contour_count']}\n")
                
                messagebox.showinfo("成功", f"结果已保存到: {file_path}")
                
            except Exception as e:
                messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def run(self):
        """运行GUI应用"""
        self.root.mainloop()

def main():
    """主函数"""
    try:
        app = ToothAnalysisGUI()
        app.run()
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        messagebox.showerror("错误", f"应用启动失败: {str(e)}")

if __name__ == "__main__":
    main()
