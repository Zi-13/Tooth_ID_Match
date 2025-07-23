import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import lstsq
import matplotlib
from scipy.spatial.distance import euclidean
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging
import json
import sqlite3
import os
from datetime import datetime
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname=r"C:\Windows\Fonts\msyh.ttc")
import traceback

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 修改字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']  # 优先黑体、雅黑
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示
plt.rcParams['font.size'] = 10

# 路径配置
CURRENT_DIR = Path(__file__).parent
IMAGES_DIR = CURRENT_DIR / 'images'
DEFAULT_IMAGE_NAME = 'TOOTH_BLUE_003.png'  # 可以轻松修改默认图片
PHOTO_PATH = str(IMAGES_DIR / DEFAULT_IMAGE_NAME)

# 验证路径是否存在
if not IMAGES_DIR.exists():
    print(f"⚠️ 图像目录不存在: {IMAGES_DIR}")
    print("💡 请创建 images 目录并放入图片")

if not Path(PHOTO_PATH).exists():
    print(f"⚠️ 默认图片不存在: {PHOTO_PATH}")
    # 尝试找到第一个可用的圖片
    image_files = list(IMAGES_DIR.glob('*.png')) + list(IMAGES_DIR.glob('*.jpg'))
    if image_files:
        PHOTO_PATH = str(image_files[0])
        print(f"💡 使用第一个找到的图片: {PHOTO_PATH}")

# 配置常量
class Config:
    DEFAULT_HSV_TOLERANCE = {'h': 15, 's': 60, 'v': 60}
    FOURIER_ORDER = 80
    MIN_CONTOUR_POINTS = 20
    SIMILARITY_THRESHOLD = 0.99  # 改为1.0作为临界值
    SIZE_TOLERANCE = 0.3
    DATABASE_PATH = "tooth_templates.db"
    TEMPLATES_DIR = "templates"
  
class FourierAnalyzer:
    """傅里叶级数分析器"""
    
    @staticmethod
    def fit_fourier_series(data: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
        """拟合傅里叶级数"""
        try:
            A = np.ones((len(t), 2 * order + 1))
            for k in range(1, order + 1):
                A[:, 2 * k - 1] = np.cos(k * t)
                A[:, 2 * k] = np.sin(k * t)
            coeffs, _, _, _ = lstsq(A, data, rcond=None)
            return coeffs
        except Exception as e:
            logger.error(f"傅里叶级数拟合失败: {e}")
            return np.zeros(2 * order + 1)

    @staticmethod
    def evaluate_fourier_series(coeffs: np.ndarray, t: np.ndarray, order: int) -> np.ndarray:
        """计算傅里叶级数值"""
        A = np.ones((len(t), 2 * order + 1))
        for k in range(1, order + 1):
            A[:, 2 * k - 1] = np.cos(k * t)
            A[:, 2 * k] = np.sin(k * t)
        return A @ coeffs

    def analyze_contour(self, points: np.ndarray, order: int = Config.FOURIER_ORDER, 
                       center_normalize: bool = True) -> dict:
        """分析轮廓的傅里叶特征"""
        try:
            x = points[:, 0].astype(float)
            y = points[:, 1].astype(float)
            
            # TODO 计算几何中心
            center_x = np.mean(x)
            center_y = np.mean(y)
            
            if center_normalize:
                # TODO 以几何中心为原点进行归一化
                x_normalized = x - center_x
                y_normalized = y - center_y
                
                # TODO 计算缩放因子（使用最大距离进行归一化）
                max_dist = np.max(np.sqrt(x_normalized**2 + y_normalized**2))
                if max_dist > 0:
                    x_normalized /= max_dist
                    y_normalized /= max_dist
            else:
                x_normalized = x
                y_normalized = y
                max_dist = 1.0
            
            N = len(points)
            t = np.linspace(0, 2 * np.pi, N)
            
            # TODO 对归一化后的坐标进行傅里叶拟合
            coeffs_x = self.fit_fourier_series(x_normalized, t, order)
            coeffs_y = self.fit_fourier_series(y_normalized, t, order)
            
            # TODO 生成更密集的参数点用于平滑显示
            t_dense = np.linspace(0, 2 * np.pi, N * 4)
            x_fit_normalized = self.evaluate_fourier_series(coeffs_x, t_dense, order)
            y_fit_normalized = self.evaluate_fourier_series(coeffs_y, t_dense, order)
            
            if center_normalize:
                # TODO 将拟合结果还原到原始坐标系
                x_fit = x_fit_normalized * max_dist + center_x
                y_fit = y_fit_normalized * max_dist + center_y
            else:
                x_fit = x_fit_normalized
                y_fit = y_fit_normalized
            
            return {
                'coeffs_x': coeffs_x,
                'coeffs_y': coeffs_y,
                'center_x': center_x,
                'center_y': center_y,
                'max_dist': max_dist,
                'order': order,
                'x_fit': x_fit,
                'y_fit': y_fit,
                'original_points': (x, y)
            }
            
        except Exception as e:
            logger.error(f"傅里叶分析失败: {e}")
            return {}

class ContourFeatureExtractor:
    """轮廓特征提取器"""
    
    def __init__(self):
        self.fourier_analyzer = FourierAnalyzer()
    
    def extract_geometric_contours(self, contour: np.ndarray, image_shape=None) -> dict:
        contours = {}
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if image_shape is not None:
            h, w = image_shape[:2]
            diag = (h**2 + w**2) ** 0.5
            area_norm = area / (diag ** 2)
            perimeter_norm = perimeter / diag
        else:
            area_norm = area
            perimeter_norm = perimeter
        x, y, w_box, h_box = cv2.boundingRect(contour)
        aspect_ratio = w_box / h_box if h_box != 0 else 0
        circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter != 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area != 0 else 0
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        corner_count = len(approx)
        contours.update({
            'area': area,
            'perimeter': perimeter,
            'area_norm': area_norm,
            'perimeter_norm': perimeter_norm,
            'aspect_ratio': aspect_ratio,
            'circularity': circularity,
            'solidity': solidity,
            'corner_count': corner_count,
            'bounding_rect': (x, y, w_box, h_box)
        })
        return contours
    
    def extract_hu_moments(self, contour: np.ndarray) -> np.ndarray:
        """提取Hu矩特征"""
        try:
            moments = cv2.moments(contour)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            # TODO 对数变换使其更稳定
            for i in range(len(hu_moments)):
                if hu_moments[i] != 0:
                    hu_moments[i] = -1 * np.copysign(1.0, hu_moments[i]) * np.log10(abs(hu_moments[i]))
                else:
                    hu_moments[i] = 0
            
            return hu_moments
        except Exception as e:
            logger.error(f"Hu矩计算失败: {e}")
            return np.zeros(7)
    
    def extract_fourier_descriptors(self, points: np.ndarray) -> np.ndarray:
        """提取傅里叶描述符"""
        try:
            fourier_data = self.fourier_analyzer.analyze_contour(points, center_normalize=True)
            if fourier_data is not None:
                coeffs_x = fourier_data['coeffs_x']
                coeffs_y = fourier_data['coeffs_y']
                # TODO 组合前11个系数（0阶+10阶*2）
                fourier_contours = np.concatenate([coeffs_x[:11], coeffs_y[:11]])
                return fourier_contours
            else:
                return np.zeros(22)
        except Exception as e:
            logger.error(f"傅里叶描述符提取失败: {e}")
            return np.zeros(22)
    
    def extract_all_contours(self, contour: np.ndarray, points: np.ndarray, image_shape=None) -> dict:
        contours = {}
        geometric_contours = self.extract_geometric_contours(contour, image_shape=image_shape)
        contours.update(geometric_contours)
        contours['hu_moments'] = self.extract_hu_moments(contour)
        contours['fourier_descriptors'] = self.extract_fourier_descriptors(points)
        fourier_data = self.fourier_analyzer.analyze_contour(points, center_normalize=True)
        if fourier_data is not None:
            contours['fourier_x_fit'] = fourier_data['x_fit'].tolist()
            contours['fourier_y_fit'] = fourier_data['y_fit'].tolist()
        return contours

class SimilarityCalculator:
    """相似度计算器"""
    
    @staticmethod
    def calculate_size_similarity(contours1: dict, contours2: dict) -> float:
        """计算尺寸相似度（只用原始面积和周长）"""
        area1 = contours1.get('area', 0)
        area2 = contours2.get('area', 0)
        perimeter1 = contours1.get('perimeter', 0)
        perimeter2 = contours2.get('perimeter', 0)
        # 计算面积相似度
        if area1 == 0 and area2 == 0:
            area_sim = 1.0
        elif area1 == 0 or area2 == 0:
            area_sim = 0.0
        else:
            area_ratio = min(area1, area2) / max(area1, area2)
            area_sim = area_ratio
        # 计算周长相似度
        if perimeter1 == 0 and perimeter2 == 0:
            perimeter_sim = 1.0
        elif perimeter1 == 0 or perimeter2 == 0:
            perimeter_sim = 0.0
        else:
            perimeter_ratio = min(perimeter1, perimeter2) / max(perimeter1, perimeter2)
            perimeter_sim = perimeter_ratio
        return 0.3*area_sim + 0.7*perimeter_sim
    
    @staticmethod
    def calculate_geometric_similarity(contours1: dict, contours2: dict) -> float:
        """计算几何特征相似度"""
        geometric_contours = ['circularity', 'aspect_ratio', 'solidity']
        geometric_weights = [0.2, 0.1, 0.7]
        
        geometric_sim = []
        for feat in geometric_contours:
            v1, v2 = contours1[feat], contours2[feat]
            if v1 == 0 and v2 == 0:
                sim = 1.0
            elif v1 == 0 or v2 == 0:
                sim = 0.0
            else:
                diff = abs(v1 - v2) / max(v1, v2)
                sim = max(0, 1 - diff * 1.5)
            geometric_sim.append(sim)
        
        return sum(w * s for w, s in zip(geometric_weights, geometric_sim))
    
    @staticmethod
    def calculate_hu_similarity(contours1: dict, contours2: dict) -> float:
        """计算Hu矩相似度"""
        try:
            hu1 = contours1['hu_moments']
            hu2 = contours2['hu_moments']
            hu_sim = cosine_similarity([hu1], [hu2])[0][0]
            return max(0, hu_sim)
        except Exception as e:
            logger.error(f"Hu矩相似度计算失败: {e}")
            return 0.0
    
    @staticmethod
    def calculate_fourier_similarity(contours1: dict, contours2: dict) -> float:
        """计算傅里叶描述符相似度"""
        try:
            fourier1 = contours1['fourier_descriptors']
            fourier2 = contours2['fourier_descriptors']
            fourier_sim = cosine_similarity([fourier1], [fourier2])[0][0]
            return max(0, fourier_sim)
        except Exception as e:
            logger.error(f"傅里叶相似度计算失败: {e}")
            return 0.0
    
    def compare_contours(self, contours1: dict, contours2: dict, 
                        size_tolerance: float = Config.SIZE_TOLERANCE) -> dict:
        """比较两个轮廓的相似度"""
        similarities = {}
        
        # TODO 计算各项相似度
        size_similarity = self.calculate_size_similarity(contours1, contours2)
        similarities['size'] = size_similarity
        
        # TODO 一级筛选：如果尺寸差异过大，直接返回低相似度
        if size_similarity < size_tolerance:
            similarities.update({
                'geometric': 0.0,
                'hu_moments': 0.0,
                'fourier': 0.0,
                'overall': size_similarity
            })
            return similarities
        
        # TODO 计算形状特征相似度
        geometric_sim = self.calculate_geometric_similarity(contours1, contours2)
        hu_sim = self.calculate_hu_similarity(contours1, contours2)
        fourier_sim = self.calculate_fourier_similarity(contours1, contours2)
        
        similarities.update({
            'geometric': geometric_sim,
            'hu_moments': hu_sim,
            'fourier': fourier_sim
        })
        
        # TODO 计算最终相似度
        shape_weights = {
            'geometric': 0.55,
            'hu_moments': 0.05,
            'fourier': 0.4
        }
        
        shape_similarity = sum(shape_weights[k] * similarities[k] for k in shape_weights)
        
        # TODO 最终相似度 = 尺寸相似度 × 形状相似度
        size_weight, shape_weight = 0.1, 0.9
        similarities['overall'] = size_similarity * size_weight + shape_similarity * shape_weight
        
        return similarities

    @staticmethod
    def compare_contours_approx(contours1: dict, contours2: dict, rel_tol=0.01, abs_tol=0.1) -> dict:
        # 主特征用相对误差
        keys = ['area', 'perimeter', 'aspect_ratio', 'circularity', 'solidity']
        all_close = True
        for k in keys:
            v1 = float(contours1.get(k, 0))
            v2 = float(contours2.get(k, 0))
            if abs(v1 - v2) / (abs(v1) + 1e-6) > rel_tol:
                all_close = False
                break
        # Hu矩、傅里叶用绝对误差
        hu1 = np.array(contours1.get('hu_moments', []))
        hu2 = np.array(contours2.get('hu_moments', []))
        if hu1.shape == hu2.shape and np.all(np.abs(hu1 - hu2) < abs_tol):
            pass
        else:
            all_close = False
        f1 = np.array(contours1.get('fourier_descriptors', []))
        f2 = np.array(contours2.get('fourier_descriptors', []))
        if f1.shape == f2.shape and np.all(np.abs(f1 - f2) < abs_tol):
            pass
        else:
            all_close = False
        if all_close:
            return {'overall': 1.0, 'size': 1.0, 'geometric': 1.0, 'hu_moments': 1.0, 'fourier': 1.0}
        # 否则走原有逻辑
        return SimilarityCalculator().compare_contours(contours1, contours2)

class DatabaseInterface:
    """数据库接口类"""
    
    def __init__(self, database_path=Config.DATABASE_PATH):
        self.database_path = database_path
        self.templates_dir = Path(Config.TEMPLATES_DIR)
    
    def load_all_templates(self):
        """加载所有模板数据"""
        if not Path(self.database_path).exists():
            logger.warning(f"数据库文件不存在: {self.database_path}")
            return {}
        
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # 检查是否有增强的特征列
            cursor.execute("PRAGMA table_info(templates)")
            columns = [column[1] for column in cursor.fetchall()]
            has_contours = 'contours_json' in columns
            
            if has_contours:
                # 使用增强的数据库结构
                cursor.execute('''
                    SELECT tooth_id, contour_file, contours_json, geometric_weights, 
                           similarity_weights, num_contours, total_area
                    FROM templates WHERE contours_json IS NOT NULL
                ''')
                
                templates = {}
                for row in cursor.fetchall():
                    tooth_id, contour_file, contours_json, geo_weights, sim_weights, num_contours, total_area = row
                    
                    # 解析特征数据
                    contours_data = json.loads(contours_json) if contours_json else []
                    
                    # 转换为match.py兼容格式
                    compatible_contours = []
                    for feature in contours_data:
                        converted = self._convert_to_match_format(feature)
                        compatible_contours.append(converted)
                    
                    templates[tooth_id] = {
                        'contours': compatible_contours,
                        'contour_file': contour_file,
                        'num_contours': num_contours,
                        'total_area': total_area,
                        'geometric_weights': json.loads(geo_weights) if geo_weights else None,
                        'similarity_weights': json.loads(sim_weights) if sim_weights else None
                    }
                
            else:
                # 使用基础数据库结构，从文件加载特征
                cursor.execute('''
                    SELECT tooth_id, contour_file, num_contours, total_area
                    FROM templates
                ''')
                
                templates = {}
                for tooth_id, contour_file, num_contours, total_area in cursor.fetchall():
                    # 尝试加载特征文件
                    contours = self._load_contours_from_file(tooth_id)
                    if contours:
                        templates[tooth_id] = {
                            'contours': contours,
                            'contour_file': contour_file,
                            'num_contours': num_contours,
                            'total_area': total_area
                        }
            
            logger.info(f"📚 已加载 {len(templates)} 个模板，共 {sum(len(t['contours']) for t in templates.values())} 个轮廓特征")
            return templates
            
        except Exception as e:
            logger.error(f"❌ 加载模板失败: {e}")
            return {}
        finally:
            conn.close()
    
    def _convert_to_match_format(self, contour_dict):
        """将单个contour字典转换为match.py兼容格式"""
        features = contour_dict['features']
        return {
            'area': features['area'],
            'perimeter': features['perimeter'],
            'aspect_ratio': features['aspect_ratio'],
            'circularity': features['circularity'],
            'solidity': features['solidity'],
            'corner_count': features['corner_count'],
            'hu_moments': np.array(features['hu_moments']),
            'fourier_descriptors': np.array(features['fourier_descriptors'])
        }
    
    def _load_contours_from_file(self, tooth_id):
        """从特征文件加载特征"""
        contours_file = self.templates_dir / "contours" / f"{tooth_id}.json"
        
        if not contours_file.exists():
            logger.warning(f"特征文件不存在: {contours_file}")
            return []
        
        try:
            with open(contours_file, 'r', encoding='utf-8') as f:
                contours_data = json.load(f)
            
            compatible_contours = []
            for contour in contours_data['contours']:
                converted = self._convert_to_match_format(contour)
                compatible_contours.append(converted)
            
            return compatible_contours
            
        except Exception as e:
            logger.error(f"❌ 加载特征文件失败: {e}")
            return []
    
    def save_match_result(self, template_id, query_image_path, query_contour_idx, similarities):
        """保存匹配结果到数据库"""
        conn = sqlite3.connect(self.database_path)
        cursor = conn.cursor()
        
        try:
            # 检查是否有匹配记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS match_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    template_id TEXT,
                    query_image_path TEXT,
                    query_contour_idx INTEGER,
                    similarity_overall REAL,
                    similarity_size REAL,
                    similarity_geometric REAL,
                    similarity_hu REAL,
                    similarity_fourier REAL,
                    match_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                INSERT INTO match_records 
                (template_id, query_image_path, query_contour_idx, 
                 similarity_overall, similarity_size, similarity_geometric, 
                 similarity_hu, similarity_fourier)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                template_id, query_image_path, query_contour_idx,
                similarities['overall'], similarities['size'], similarities['geometric'],
                similarities['hu_moments'], similarities['fourier']
            ))
            conn.commit()
            
        except Exception as e:
            logger.error(f"❌ 保存匹配结果失败: {e}")
        finally:
            conn.close()

def load_features_templates(features_dir="templates/features"):
    templates = {}
    for fname in os.listdir(features_dir):
        if fname.endswith("_features.json"):
            tooth_id = fname.split("_features.json")[0].upper()
            with open(os.path.join(features_dir, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
            templates[tooth_id] = data["features"]
    return templates

class ToothMatcher:
    """牙齿匹配器主类 - 增强版"""
    
    def __init__(self):
        self.feature_extractor = ContourFeatureExtractor()
        self.similarity_calculator = SimilarityCalculator()
        self.fourier_analyzer = FourierAnalyzer()
        self.db_interface = DatabaseInterface()
        self.templates = load_features_templates()
        self.current_image_path = None
        self.highlight_template = None  # (template_id, template_contour_idx)
        self._db_match_line_boxes = []  # 存储匹配区每行的bbox和match_id
        self.match_highlight_idx = None  # 当前色块下高亮的数据库匹配索引

    def load_templates(self):
        """加载模板库"""
        self.templates = load_features_templates()
        return len(self.templates) > 0
    
    def match_against_database(self, query_features_list, threshold=Config.SIMILARITY_THRESHOLD):
        """与数据库模板进行匹配"""
        if not self.templates:
            logger.warning("❌ 未加载模板数据，请先使用 BuildTheLab 创建模板")
            return {}
        all_matches = {}
        for query_idx, query_features in enumerate(query_features_list):
            query_matches = []
            for template_id, template_features_list in self.templates.items():
                for template_idx, template_features in enumerate(template_features_list):
                    similarities = self.similarity_calculator.compare_contours_approx(
                        query_features, template_features, rel_tol=0.01, abs_tol=0.1)
                    # 删除详细调试输出
                    if similarities['overall'] >= threshold:
                        match_info = {
                            'template_id': template_id,
                            'template_contour_idx': template_idx,
                            'similarity': similarities['overall'],
                            'details': similarities,
                            'query_contour_idx': query_idx
                        }
                        query_matches.append(match_info)
                        # 保存匹配结果到数据库
                        if self.current_image_path:
                            self.db_interface.save_match_result(
                                template_id, self.current_image_path, query_idx, similarities
                            )
            # 按相似度排序
            query_matches.sort(key=lambda x: x['similarity'], reverse=True)
            all_matches[f'query_{query_idx}'] = query_matches
        return all_matches
    
    def find_similar_contours(self, target_contours: dict, all_contours: list, 
                             threshold: float = Config.SIMILARITY_THRESHOLD,
                             size_tolerance: float = Config.SIZE_TOLERANCE) -> list:
        """找到与目标轮廓相似的所有轮廓（当前图像内部）"""
        similar_contours = []
        
        for i, contours in enumerate(all_contours):
            if contours == target_contours:
                continue
            
            similarities = self.similarity_calculator.compare_contours(
                target_contours, contours, size_tolerance)
            
            if similarities['overall'] >= threshold:
                similar_contours.append({
                    'index': i,
                    'similarity': similarities['overall'],
                    'details': similarities
                })
        
        similar_contours.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_contours
    
    def process_image(self, image_path: str):
        """处理图像的主函数"""
        self.current_image_path = image_path
        
        # 验证文件路径
        if not Path(image_path).exists():
            logger.error(f"图像文件不存在: {image_path}")
            return
        
        # 加载模板库
        if not self.load_templates():
            logger.warning("⚠️ 未找到模板库，仅显示当前图像轮廓分析")
        
        img = cv2.imread(image_path)
        if img is None:
            logger.error("图片读取失败")
            return
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        picked_colors = self._pick_colors(img, hsv)
        
        if not picked_colors:
            logger.warning("未选取颜色")
            return
        
        # 创建掩码并提取轮廓
        mask = self._create_mask(hsv, picked_colors)
        color_extract = cv2.bitwise_and(img, img, mask=mask)
        
        # 处理轮廓
        valid_contours, all_contours = self._process_contours(mask)
        
        if not valid_contours:
            logger.warning("未检测到有效轮廓")
            return
        
        logger.info(f"检测到 {len(valid_contours)} 个有效轮廓")
        
        # 修正 query_features_list 的生成方式
        query_features_list = [c['contours'] for c in valid_contours]
        matches = self.match_against_database(query_features_list)
        
        # 显示交互式界面
        self._show_interactive_display(color_extract, valid_contours, all_contours, matches)
    
    def _pick_colors(self, img: np.ndarray, hsv: np.ndarray) -> list:
        """颜色选择 - 自动调整显示大小"""
        picked = []
        original_img = img.copy()
        original_hsv = hsv.copy()
        
        # 获取屏幕尺寸的估计值（保守估计）
        max_width = 1200
        max_height = 800
        
        # 计算缩放比例
        h, w = img.shape[:2]
        scale_w = max_width / w if w > max_width else 1.0
        scale_h = max_height / h if h > max_height else 1.0
        scale = min(scale_w, scale_h)
        
        # 如果需要缩放，则缩放图像
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            display_img = cv2.resize(img, (new_w, new_h))
            display_hsv = cv2.resize(hsv, (new_w, new_h))
            logger.info(f"图像缩放: {w}x{h} -> {new_w}x{new_h} (缩放比例: {scale:.2f})")
        else:
            display_img = img
            display_hsv = hsv
            scale = 1.0
        
        def on_mouse(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # 将显示坐标转换回原始图像坐标
                orig_x = int(x / scale)
                orig_y = int(y / scale)
                
                # 确保坐标在原始图像范围内
                orig_x = max(0, min(orig_x, original_img.shape[1] - 1))
                orig_y = max(0, min(orig_y, original_img.shape[0] - 1))
                
                color = original_hsv[orig_y, orig_x]
                logger.info(f"选中点 显示坐标:({x},{y}) -> 原始坐标:({orig_x},{orig_y}) HSV: {color}")
                picked.append(color)
                
                # 在图像上标记选中点
                cv2.circle(display_img, (x, y), 5, (0, 255, 0), 2)
                cv2.putText(display_img, f"{len(picked)}", (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("点击选取目标区域颜色 (按空格键完成选择)", display_img)
        
        # 创建窗口并设置可调整大小
        window_name = "点击选取目标区域颜色 (按空格键完成选择)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, display_img)
        cv2.setMouseCallback(window_name, on_mouse)
        
        print("🎯 颜色选择说明:")
        print("  • 点击图像中的目标区域来选择颜色")
        print("  • 可以选择多个颜色点")
        print("  • 按空格键或ESC键完成选择")
        print("  • 按R键重置")
        print("  • 按Q键取消并退出")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') or key == 27:  # 空格键或ESC键完成选择
                if picked:
                    print(f"✅ 完成选择，共选择了 {len(picked)} 个颜色点")
                    break
                else:
                    print("⚠️ 请先选择至少一个颜色点")
            elif key == ord('q') or key == ord('Q'):  # Q键取消
                print("❌ 取消颜色选择")
                picked = []
                break
            elif key == ord('r'):  # R键重置
                picked = []
                display_img = cv2.resize(img, (int(w * scale), int(h * scale))) if scale < 1.0 else img.copy()
                cv2.imshow(window_name, display_img)
                print("🔄 已重置选择")
        
        cv2.destroyAllWindows()
        
        if picked:
            print(f"✅ 颜色选择完成！已选择 {len(picked)} 个颜色点")
            # 显示选择的颜色信息
            for i, color in enumerate(picked):
                print(f"  点{i+1}: HSV({color[0]}, {color[1]}, {color[2]})")
        else:
            print("❌ 未选择任何颜色，程序将退出")
        
        return picked
    
    def _create_mask(self, hsv: np.ndarray, picked_colors: list) -> np.ndarray:
        """创建颜色掩码"""
        hsv_arr = np.array(picked_colors)
        h, s, v = np.mean(hsv_arr, axis=0).astype(int)
        logger.info(f"HSV picked: {h}, {s}, {v}")
        
        tolerance = Config.DEFAULT_HSV_TOLERANCE
        
        lower = np.array([
            max(0, h - tolerance['h']), 
            max(0, s - tolerance['s']), 
            max(0, v - tolerance['v']-10)
        ])
        upper = np.array([
            min(179, h + tolerance['h']), 
            min(255, s + tolerance['s']+10), 
            min(255, v + tolerance['v'])
        ])
        
        logger.info(f"HSV范围 - lower: {lower}, upper: {upper}")
        return cv2.inRange(hsv, lower, upper)
    
    def _process_contours(self, mask: np.ndarray) -> tuple:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        valid_contours = []
        all_contours = []
        areas = [cv2.contourArea(c) for c in contours]
        if areas:
            max_area = max(areas)
            min_area = min(areas)
            if max_area > 0 and max_area / max(min_area, 1e-6) > 100:
                area_threshold = max_area / 100
                filtered = [(i, c) for i, c in enumerate(contours) if cv2.contourArea(c) >= area_threshold]
                contours = [c for i, c in filtered]
        # 获取图像shape
        image_shape = None
        if hasattr(self, 'current_image_path') and self.current_image_path is not None:
            img = cv2.imread(self.current_image_path)
            if img is not None:
                image_shape = img.shape
        for i, contour in enumerate(contours):
            if contour.shape[0] < Config.MIN_CONTOUR_POINTS:
                continue
            area = cv2.contourArea(contour)
            length = cv2.arcLength(contour, True)
            points = contour[:, 0, :]
            contours = self.feature_extractor.extract_all_contours(contour, points, image_shape=image_shape)
            # if i == 1:  # 假设你要比对第1个色块
            #      print("【调试】当前色块特征：", contours)
            valid_contours.append({
                'contour': contour,
                'points': points,
                'area': area,
                'length': length,
                'idx': i,
                'contours': contours
            })
            all_contours.append(contours)
        return valid_contours, all_contours
    
    def _show_interactive_display(self, color_extract: np.ndarray, 
                             valid_contours: list, all_contours: list, matches):
        n_contours = len(valid_contours)
        linewidth = max(0.5, 2 - 0.03 * n_contours)
        show_legend = n_contours <= 15
        
        # 调整布局：删除色块放大视图，放大模板原图预览
        if self.templates:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))  # 改为2x3布局
            # 重新分配子图
            ax_img, ax_fit, ax_template_preview = axes[0]  # 上排：颜色提取、轮廓显示、模板原图预览(放大)
            ax_db_matches, ax_stats, ax_history = axes[1]  # 下排：数据库匹配、统计、历史
        else:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # 改为1x3布局
            ax_img, ax_fit, ax_template_preview = axes
            ax_db_matches = ax_stats = ax_history = None
        
        # 设置各子图标题
        ax_img.set_title("颜色提取结果", fontproperties=myfont)
        ax_img.imshow(cv2.cvtColor(color_extract, cv2.COLOR_BGR2RGB))
        ax_img.axis('off')
        
        ax_fit.set_title("轮廓显示", fontproperties=myfont)
        ax_fit.axis('equal')
        ax_fit.invert_yaxis()
        ax_fit.grid(True)
        
        # 放大的模板原图预览区
        ax_template_preview.set_title("模板原图预览", fontproperties=myfont, fontsize=14)
        ax_template_preview.axis('off')
        
        # 初始化数据库匹配信息
        if self.templates:
            if ax_db_matches is not None:
                ax_db_matches.set_title("数据库匹配结果", fontproperties=myfont)
                ax_db_matches.axis('off')
            if ax_stats is not None:
                ax_stats.set_title("模板库统计", fontproperties=myfont)
                ax_stats.axis('off')
            if ax_history is not None:
                ax_history.set_title("匹配历史", fontproperties=myfont)
                ax_history.axis('off')
            
            # 显示模板库统计
            total_templates = len(self.templates)
            total_contours = sum(len(t) for t in self.templates.values())
            stats_text = f"模板库统计:\n总模板数: {total_templates}\n总轮廓数: {total_contours}\n\n"
            stats_text += "模板列表:\n"
            for i, (template_id, data) in enumerate(list(self.templates.items())[:10]):
                stats_text += f"{i+1}. {template_id} ({len(data)}个轮廓)\n"
            if total_templates > 10:
                stats_text += f"... 还有 {total_templates-10} 个模板"
            if ax_stats is not None:
                ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                             fontsize=10, verticalalignment='top', fontproperties=myfont)
        
        selected_idx = [0]
        self.match_highlight_idx = None
        self.highlight_template = None
        
        def draw_all(highlight_idx=None):
            print("[DRAW_ALL] 调用栈:")
            print("[DRAW_ALL] 当前高亮模板:", self.highlight_template)
            print("[DRAW_ALL] 当前match_highlight_idx:", self.match_highlight_idx)
            print("[DRAW_ALL] matches keys:", list(matches.keys()))
            key = f'query_{highlight_idx}'
            print("[DRAW_ALL] 当前色块key:", key, "匹配列表长度:", len(matches.get(key, [])))
            
            # 更新轮廓显示（移除ax_zoom参数）
            self._draw_contours_enhanced(ax_fit, valid_contours, all_contours, 
                                       highlight_idx, linewidth, show_legend, fig,
                                       ax_db_matches if self.templates else None, matches)
            
            # 更新放大的模板原图预览区
            ax_template_preview.clear()
            ax_template_preview.set_title("模板原图预览", fontproperties=myfont, fontsize=14)
            ax_template_preview.axis('off')
            
            if self.highlight_template is not None and self.templates:
                template_id, template_contour_idx = self.highlight_template
                print("[DRAW_ALL] 模板原图区高亮分支:", template_id, template_contour_idx)
                
                # 加载原图
                img_path = f"templates/images/{template_id}.png"
                print("[DRAW_ALL] 原图路径:", img_path)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax_template_preview.imshow(img_rgb)
                    
                    # 加载轮廓点
                    contour_json = f"templates/contours/{template_id}.json"
                    print("[DRAW_ALL] 轮廓json路径:", contour_json)
                    if os.path.exists(contour_json):
                        with open(contour_json, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if 'contours' in data and 0 <= template_contour_idx < len(data['contours']):
                            points = np.array(data['contours'][template_contour_idx]['points'])
                            print("[DRAW_ALL] points.shape:", points.shape)
                            try:
                                # 使用更明显的高亮效果
                                ax_template_preview.fill(points[:,0], points[:,1], 
                                                       color='red', alpha=0.6, zorder=10, 
                                                       label=f'匹配轮廓: {template_contour_idx+1}')
                                ax_template_preview.plot(points[:,0], points[:,1], 
                                                       color='darkred', linewidth=3, zorder=11)
                                print("[DRAW_ALL] 轮廓绘制完成")
                                
                                # 添加标注信息（黑色字体，无背景框）
                                center_x, center_y = np.mean(points, axis=0)
                                ax_template_preview.text(center_x, center_y, str(template_contour_idx+1), 
                                                       fontsize=16, fontweight='bold', 
                                                       color='black', ha='center', va='center', 
                                                       zorder=12)
                                
                            except Exception as e:
                                print("[DRAW_ALL] 轮廓绘制异常:", e)
                        else:
                            print("[DRAW_ALL] 轮廓点索引超界或无contours")
                    else:
                        print("[DRAW_ALL] 轮廓json文件不存在")
                    
                    # 添加模板信息标题
                    info_text = f"模板: {template_id}\n轮廓: {template_contour_idx+1}"
                    ax_template_preview.text(0.02, 0.98, info_text, 
                                           transform=ax_template_preview.transAxes,
                                           fontsize=12, fontweight='bold', color='blue',
                                           ha='left', va='top', fontproperties=myfont,
                                           bbox=dict(facecolor='white', alpha=0.8, 
                                                   edgecolor='blue', boxstyle='round,pad=0.3'))
                else:
                    print(f"[DRAW_ALL] 未找到原图: {img_path}")
                    ax_template_preview.text(0.5, 0.5, f"未找到模板图像\n{template_id}", 
                                           ha='center', va='center', fontsize=14, color='red',
                                           fontproperties=myfont)
            else:
                print("[DRAW_ALL] 无模板高亮分支")
                # 显示使用说明
                help_text = ("🦷 模板原图预览区\n\n"
                            "📖 使用方法:\n"
                            "• ←→ 切换色块\n"
                            "• ↓ 选择匹配项\n"
                            "• 点击匹配项查看模板\n\n"
                            "💡 此区域将显示匹配到的\n"
                            "模板原始图像和轮廓位置")
                ax_template_preview.text(0.5, 0.5, help_text, ha='center', va='center', 
                                       fontsize=12, color='gray', fontproperties=myfont,
                                       bbox=dict(facecolor='lightgray', alpha=0.3, 
                                               boxstyle='round,pad=0.5'))
            
            fig.canvas.draw_idle()
        
        def on_key(event):
            print(f"[ON_KEY] 按键: {event.key}, 当前选中色块: {selected_idx[0]}, match_highlight_idx: {self.match_highlight_idx}")
            
            if event.key == 'right':
                selected_idx[0] = (selected_idx[0] + 1) % n_contours
                self.match_highlight_idx = None
                self.highlight_template = None
                print(f"[ON_KEY] 切换到色块 {selected_idx[0]}")
                draw_all(highlight_idx=selected_idx[0])
                
            elif event.key == 'left':
                selected_idx[0] = (selected_idx[0] - 1) % n_contours
                self.match_highlight_idx = None
                self.highlight_template = None
                print(f"[ON_KEY] 切换到色块 {selected_idx[0]}")
                draw_all(highlight_idx=selected_idx[0])
                
            elif event.key in ['escape', 'up']:
                if self.match_highlight_idx is not None or self.highlight_template is not None:
                    self.match_highlight_idx = None
                    self.highlight_template = None
                    print("[ON_KEY] 取消匹配高亮，返回色块高亮")
                    draw_all(highlight_idx=selected_idx[0])
                
            elif event.key == 'down':
                current_key = f'query_{selected_idx[0]}'
                match_list = matches.get(current_key, [])
                
                if not match_list:
                    print(f"[ON_KEY] 色块 {selected_idx[0]} 无匹配项")
                    return
                
                if self.match_highlight_idx is None:
                    self.match_highlight_idx = 0
                    print(f"[ON_KEY] 选中第一个匹配项 (索引0)")
                else:
                    self.match_highlight_idx = (self.match_highlight_idx + 1) % len(match_list)
                    print(f"[ON_KEY] 切换到匹配项 {self.match_highlight_idx}")
                
                if 0 <= self.match_highlight_idx < len(match_list):
                    match = match_list[self.match_highlight_idx]
                    self.highlight_template = (match['template_id'], match['template_contour_idx'])
                    print(f"[ON_KEY] 设置高亮模板: {self.highlight_template}")
                
                draw_all(highlight_idx=selected_idx[0])
            
            elif event.key == 'q':
                print("[ON_KEY] 退出程序")
                plt.close()
            
            else:
                print(f"[ON_KEY] 未处理的按键: {event.key}")
        
        def on_db_match_click(event):
            if ax_db_matches is None or event.inaxes != ax_db_matches:
                return
            
            if not hasattr(self, '_db_match_line_boxes') or not self._db_match_line_boxes:
                return
            
            click_x, click_y = event.xdata, event.ydata
            if click_x is None or click_y is None:
                return
            
            for idx, (bbox, match_id) in enumerate(self._db_match_line_boxes):
                x0, y0, x1, y1 = bbox
                if x0 <= click_x <= x1 and y0 <= click_y <= y1:
                    self.highlight_template = match_id
                    self.match_highlight_idx = idx
                    draw_all(highlight_idx=selected_idx[0])
                    return
        
        def on_click(event):
            # 更新点击检测，移除ax_zoom相关判断
            if self.templates and event.inaxes not in [ax_img, ax_fit, ax_db_matches]:
                pass
            on_db_match_click(event)
        
        # 绑定事件
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        
        # 初始显示
        draw_all(highlight_idx=selected_idx[0])
        
        plt.tight_layout()
        plt.show()
    
    def _draw_contours_enhanced(self, ax, valid_contours, all_contours, highlight_idx, 
                               linewidth, show_legend, fig, ax_db_matches, matches):
        """增强版轮廓绘制方法"""
        ax.clear()
        ax.set_title("轮廓显示", fontproperties=myfont)
        ax.axis('equal')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        
        colors = plt.cm.tab20(np.linspace(0, 1, len(valid_contours)))
        
        # 绘制所有轮廓
        for i, contour_info in enumerate(valid_contours):
            contour = contour_info['contour']
            points = contour_info['points']
            area = contour_info['area']
            
            color = colors[i]
            alpha = 0.6 if i == highlight_idx else 0.4
            edge_alpha = 1.0 if i == highlight_idx else 0.7
            linewidth_current = linewidth * 3 if i == highlight_idx else linewidth * 2
            
            # 绘制填充轮廓（类似您图片中的效果）
            ax.fill(points[:, 0], points[:, 1], color=color, 
                   alpha=alpha, label=f'色块 {i+1} (面积:{area:.0f})')
            
            # 绘制轮廓边框
            ax.plot(points[:, 0], points[:, 1], color=color, 
                   linewidth=linewidth_current, alpha=edge_alpha)
            
            # 标注色块编号（黑色字体，无背景框）
            center = np.mean(points, axis=0)
            ax.text(center[0], center[1], str(i+1), 
                   fontsize=10, ha='center', va='center', 
                   fontweight='bold', color='black')
        
        # 高亮显示匹配模板轮廓（如果存在）
        if self.highlight_template and highlight_idx is not None:
            template_id, template_contour_idx = self.highlight_template
            
            # 在轮廓图上添加匹配指示
            if highlight_idx < len(valid_contours):
                contour_info = valid_contours[highlight_idx]
                points = contour_info['points']
                center = np.mean(points, axis=0)
                
                # 添加匹配指示标记（纯红圆点）
                ax.plot(center[0], center[1], 'o', markersize=2, 
                       color='red', 
                       label=f'匹配: {template_id}-{template_contour_idx+1}')
        
        if show_legend:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), 
                     fontsize=8, prop=myfont)
        
        # 更新数据库匹配显示
        if ax_db_matches is not None and matches:
            self._update_db_matches_display(ax_db_matches, matches, highlight_idx)
    
    def _update_db_matches_display(self, ax, matches, highlight_idx):
        """更新数据库匹配显示"""
        ax.clear()
        ax.set_title("数据库匹配结果", fontproperties=myfont)
        ax.axis('off')
        
        if highlight_idx is None:
            ax.text(0.5, 0.5, "请选择一个色块查看匹配结果", 
                   ha='center', va='center', fontproperties=myfont,
                   transform=ax.transAxes)
            return
        
        query_key = f'query_{highlight_idx}'
        query_matches = matches.get(query_key, [])
        
        if not query_matches:
            ax.text(0.5, 0.5, f"色块 {highlight_idx+1} 无匹配结果", 
                   ha='center', va='center', fontproperties=myfont,
                   transform=ax.transAxes, color='red')
            return
        
        # 显示匹配结果
        y_pos = 0.95
        line_height = 0.08
        
        ax.text(0.05, y_pos, f"色块 {highlight_idx+1} 的匹配结果:", 
               fontsize=14, fontweight='bold', fontproperties=myfont,
               transform=ax.transAxes)
        y_pos -= line_height
        
        self._db_match_line_boxes = []  # 重置点击区域
        
        for i, match in enumerate(query_matches[:10]):  # 最多显示10个匹配
            similarity = match['similarity']
            template_id = match['template_id']
            template_idx = match['template_contour_idx']
            
            # 高亮当前选中的匹配项
            if i == self.match_highlight_idx:
                bg_color = 'yellow'
                text_color = 'black'
                alpha = 0.8
            else:
                bg_color = 'lightblue' if i % 2 == 0 else 'white'
                text_color = 'black'
                alpha = 0.3
            
            # 添加背景框
            bbox = dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=alpha)
            
            match_text = f"{i+1}. {template_id}-{template_idx+1}: {similarity:.3f}"
            text_obj = ax.text(0.05, y_pos, match_text, 
                              fontsize=10, fontproperties=myfont,
                              transform=ax.transAxes, color=text_color,
                              bbox=bbox)
            
            # 记录点击区域
            bbox_coords = (0.05, y_pos - line_height/2, 0.95, y_pos + line_height/2)
            match_id = (template_id, template_idx)
            self._db_match_line_boxes.append((bbox_coords, match_id))
            
            y_pos -= line_height
            
            if y_pos < 0.1:  # 避免超出显示区域
                break


if __name__ == "__main__":
    """主执行入口"""
    import sys
    
    # 检查是否提供了图像路径参数
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # 使用默认图像
        image_path = PHOTO_PATH
    
    print(f"🦷 牙齿匹配系统启动")
    print(f"📸 图像路径: {image_path}")
    
    # 创建匹配器并处理图像
    matcher = ToothMatcher()
    
    try:
        matcher.process_image(image_path)
        print("✅ 处理完成")
    except Exception as e:
        print(f"❌ 处理失败: {e}")
        traceback.print_exc()
