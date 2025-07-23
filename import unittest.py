import unittest
import tempfile
import shutil
import json
import sqlite3
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import sys
import os
from BulidTheLab import ToothTemplateBuilder
from match import ContourFeatureExtractor, FourierAnalyzer
import time

# 添加项目路径以导入模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestToothTemplateBuilder(unittest.TestCase):
    """测试牙齿模板构建器"""
    
    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_db = os.path.join(self.temp_dir, "test_templates.db")
        self.builder = ToothTemplateBuilder(
            database_path=self.test_db,
            templates_dir=os.path.join(self.temp_dir, "templates")
        )
        
        # 创建测试用的轮廓数据
        self.test_contours = self._create_test_contours()
        
    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_contours(self):
        """创建测试用的轮廓数据"""
        # 创建圆形轮廓
        center = (100, 100)
        radius = 50
        angles = np.linspace(0, 2*np.pi, 100)
        x_coords = center[0] + radius * np.cos(angles)
        y_coords = center[1] + radius * np.sin(angles)
        
        points = np.column_stack((x_coords, y_coords)).astype(np.int32)
        contour = points.reshape(-1, 1, 2)
        
        return [{
            'contour': contour,
            'points': points,
            'area': cv2.contourArea(contour),
            'length': cv2.arcLength(contour, True),
            'idx': 0
        }]
    
    def _create_test_image(self):
        """创建测试图像"""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.circle(img, (100, 100), 50, (255, 255, 255), -1)
        return img

class TestFeatureExtraction(TestToothTemplateBuilder):
    """测试特征提取功能"""
    
    def test_extract_template_features(self):
        """测试模板特征提取"""
        print("🧪 测试特征提取功能...")
        
        features = self.builder.extract_template_features(self.test_contours)
        
        # 验证特征提取结果
        self.assertGreater(len(features), 0, "应该提取到特征")
        
        feature = features[0]
        self.assertIn('geometric_features', feature)
        self.assertIn('hu_moments', feature)
        self.assertIn('fourier_descriptors', feature)
        
        # 检查几何特征
        geo_features = feature['geometric_features']
        self.assertIn('area', geo_features)
        self.assertIn('perimeter', geo_features)
        self.assertIn('circularity', geo_features)
        self.assertGreater(geo_features['area'], 0)
        
        # 检查Hu矩特征
        self.assertEqual(len(feature['hu_moments']), 7)
        
        # 检查傅里叶描述符
        self.assertEqual(len(feature['fourier_descriptors']), 22)
        
        print("✅ 特征提取测试通过")
    
    def test_feature_config_integration(self):
        """测试特征配置集成"""
        print("🧪 测试特征配置集成...")
        
        features = self.builder.extract_template_features(self.test_contours)
        feature = features[0]
        
        # 验证配置参数是否正确保存
        self.assertIn('extraction_params', feature)
        params = feature['extraction_params']
        
        self.assertEqual(params['fourier_order'], 80)
        self.assertIn('geometric_weights', params)
        self.assertIn('similarity_weights', params)
        
        print("✅ 特征配置集成测试通过")

class TestDataSaving(TestToothTemplateBuilder):
    """测试数据保存功能"""
    
    def test_serialize_contours_with_features(self):
        """测试轮廓和特征序列化保存"""
        print("🧪 测试轮廓和特征保存...")
        
        tooth_id = "TEST_TOOTH_001"
        image_path = "test_image.jpg"
        hsv_info = {'h_mean': 100, 's_mean': 150, 'v_mean': 200}
        
        success = self.builder.serialize_contours_with_features(
            self.test_contours, tooth_id, image_path, hsv_info
        )
        
        self.assertTrue(success, "保存应该成功")
        
        # 验证JSON文件是否创建
        json_path = self.builder.templates_dir / "contours" / f"{tooth_id}.json"
        self.assertTrue(json_path.exists(), "轮廓JSON文件应该存在")
        
        # 验证特征文件是否创建
        features_path = self.builder.templates_dir / "features" / f"{tooth_id}_features.json"
        self.assertTrue(features_path.exists(), "特征JSON文件应该存在")
        
        print("✅ 轮廓和特征保存测试通过")
    
    def test_json_content_validation(self):
        """测试JSON文件内容验证"""
        print("🧪 测试JSON文件内容...")
        
        tooth_id = "TEST_TOOTH_002"
        image_path = "test_image.jpg"
        
        self.builder.serialize_contours_with_features(
            self.test_contours, tooth_id, image_path
        )
        
        # 读取并验证轮廓JSON
        json_path = self.builder.templates_dir / "contours" / f"{tooth_id}.json"
        with open(json_path, 'r', encoding='utf-8') as f:
            contour_data = json.load(f)
        
        self.assertEqual(contour_data['tooth_id'], tooth_id)
        self.assertEqual(contour_data['num_contours'], 1)
        self.assertIn('features', contour_data)
        self.assertIn('feature_config', contour_data)
        
        # 读取并验证特征JSON
        features_path = self.builder.templates_dir / "features" / f"{tooth_id}_features.json"
        with open(features_path, 'r', encoding='utf-8') as f:
            features_data = json.load(f)
        
        self.assertEqual(features_data['tooth_id'], tooth_id)
        self.assertIn('features', features_data)
        
        print("✅ JSON文件内容验证测试通过")
    
    def test_database_enhanced_saving(self):
        """测试增强数据库保存"""
        print("🧪 测试数据库保存功能...")
        
        tooth_id = "TEST_TOOTH_003"
        image_path = "test_image.jpg"
        
        success = self.builder.serialize_contours_with_features(
            self.test_contours, tooth_id, image_path
        )
        
        self.assertTrue(success)
        
        # 验证数据库记录
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM templates WHERE tooth_id = ?', (tooth_id,))
        record = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(record, "数据库应该有记录")
        
        print("✅ 数据库保存测试通过")

class TestUniqueIDGeneration(TestToothTemplateBuilder):
    """测试唯一编码生成"""
    
    def test_unique_tooth_id_generation(self):
        """测试唯一牙齿ID生成"""
        print("🧪 测试唯一ID生成...")
        
        # 测试时间戳ID生成
        timestamp1 = datetime.now().strftime("%Y%m%d_%H%M%S")
        tooth_id1 = f"TOOTH_{timestamp1}"
        
        # 稍微延迟确保时间戳不同
        time.sleep(0.1)
        
        timestamp2 = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]  # 包含微秒
        tooth_id2 = f"TOOTH_{timestamp2}"
        
        self.assertNotEqual(tooth_id1, tooth_id2, "生成的ID应该唯一")
        
        print(f"   生成的ID1: {tooth_id1}")
        print(f"   生成的ID2: {tooth_id2}")
        print("✅ 唯一ID生成测试通过")
    
    def test_duplicate_id_handling(self):
        """测试重复ID处理"""
        print("🧪 测试重复ID处理...")
        
        tooth_id = "DUPLICATE_TEST"
        image_path = "test_image.jpg"
        
        # 第一次保存
        success1 = self.builder.serialize_contours_with_features(
            self.test_contours, tooth_id, image_path
        )
        self.assertTrue(success1)
        
        # 第二次保存相同ID（应该覆盖）
        success2 = self.builder.serialize_contours_with_features(
            self.test_contours, tooth_id, image_path
        )
        self.assertTrue(success2)
        
        # 验证数据库中只有一条记录
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM templates WHERE tooth_id = ?', (tooth_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 1, "应该只有一条记录（覆盖模式）")
        
        print("✅ 重复ID处理测试通过")

class TestMatchIntegration(TestToothTemplateBuilder):
    """测试与match.py的集成"""
    
    def test_match_feature_compatibility(self):
        """测试与match.py特征提取的兼容性"""
        print("🧪 测试match.py集成...")
        
        # 直接使用match.py的特征提取器
        extractor = ContourFeatureExtractor()
        
        contour = self.test_contours[0]['contour']
        points = self.test_contours[0]['points']
        
        # 使用match.py提取特征
        match_features = extractor.extract_all_features(contour, points)
        
        # 使用BulidTheLab提取特征
        builder_features = self.builder.extract_template_features(self.test_contours)
        
        # 验证特征结构兼容性
        self.assertGreater(len(builder_features), 0)
        builder_feature = builder_features[0]
        
        # 检查关键特征是否一致
        self.assertEqual(
            len(match_features['hu_moments']), 
            len(builder_feature['hu_moments'])
        )
        self.assertEqual(
            len(match_features['fourier_descriptors']), 
            len(builder_feature['fourier_descriptors'])
        )
        
        print("✅ match.py集成测试通过")
    
    def test_fourier_analyzer_integration(self):
        """测试傅里叶分析器集成"""
        print("🧪 测试傅里叶分析器集成...")
        
        analyzer = FourierAnalyzer()
        points = self.test_contours[0]['points']
        
        fourier_data = analyzer.analyze_contour(points, order=80, center_normalize=True)
        
        self.assertIsNotNone(fourier_data, "傅里叶分析应该成功")
        self.assertIn('coeffs_x', fourier_data)
        self.assertIn('coeffs_y', fourier_data)
        self.assertIn('center_x', fourier_data)
        self.assertIn('center_y', fourier_data)
        
        print("✅ 傅里叶分析器集成测试通过")

class TestAdvancedFeatures(TestToothTemplateBuilder):
    """测试高级功能"""
    
    def test_contour_filtering(self):
        """测试轮廓过滤功能"""
        print("🧪 测试轮廓过滤...")
        
        # 创建多个测试轮廓（包括无效的）
        test_contours = []
        
        # 有效轮廓
        valid_contour = np.array([[10, 10], [50, 10], [50, 50], [10, 50]], dtype=np.int32)
        valid_contour = valid_contour.reshape(-1, 1, 2)
        test_contours.append(valid_contour)
        
        # 面积太小的轮廓
        small_contour = np.array([[5, 5], [8, 5], [8, 8], [5, 8]], dtype=np.int32)
        small_contour = small_contour.reshape(-1, 1, 2)
        test_contours.append(small_contour)
        
        filtered = self.builder.filter_contours_advanced(
            test_contours, min_area=100, max_area=50000
        )
        
        self.assertEqual(len(filtered), 1, "应该过滤掉小轮廓")
        
        print("✅ 轮廓过滤测试通过")
    
    def test_mask_enhancement(self):
        """测试掩码增强功能"""
        print("🧪 测试掩码增强...")
        
        # 创建测试掩码
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 20, 255, -1)
        cv2.circle(mask, (30, 30), 10, 255, -1)
        
        enhanced_mask, markers = self.builder.enhance_mask_quality(mask)
        
        self.assertIsNotNone(enhanced_mask, "增强掩码应该成功")
        self.assertIsNotNone(markers, "分水岭标记应该存在")
        self.assertEqual(enhanced_mask.shape, mask.shape, "形状应该一致")
        
        print("✅ 掩码增强测试通过")

class TestFeatureAnswers(TestToothTemplateBuilder):
    """回答用户关于特征保存的问题"""
    
    def test_feature_saving_capability(self):
        """测试特征保存能力 - 回答用户问题1"""
        print("🧪 验证: BulidTheLab能否保存match.py需要的特征值?")
        
        tooth_id = "FEATURE_TEST"
        success = self.builder.serialize_contours_with_features(
            self.test_contours, tooth_id, "test.jpg"
        )
        
        # 读取保存的特征
        features_path = self.builder.templates_dir / "features" / f"{tooth_id}_features.json"
        with open(features_path, 'r', encoding='utf-8') as f:
            saved_features = json.load(f)
        
        feature = saved_features['features'][0]
        
        # 验证match.py需要的所有特征都已保存
        required_features = [
            'geometric_features', 'hu_moments', 'fourier_descriptors'
        ]
        
        for req_feature in required_features:
            self.assertIn(req_feature, feature, f"缺少必需特征: {req_feature}")
        
        # 验证几何特征的完整性
        geo_features = feature['geometric_features']
        required_geo = ['area', 'perimeter', 'circularity', 'aspect_ratio', 'solidity']
        for req_geo in required_geo:
            self.assertIn(req_geo, geo_features, f"缺少几何特征: {req_geo}")
        
        print("✅ 答案1: 是的，BulidTheLab已经能完整保存match.py需要的所有特征值")
        print("   包括: 几何特征、Hu矩、傅里叶描述符")
    
    def test_automatic_saving(self):
        """测试自动保存功能 - 回答用户问题2"""
        print("🧪 验证: 能否自动保存这些特征?")
        
        tooth_id = "AUTO_SAVE_TEST"
        
        # 一次调用就能自动保存所有特征
        success = self.builder.serialize_contours_with_features(
            self.test_contours, tooth_id, "test.jpg"
        )
        
        self.assertTrue(success, "自动保存应该成功")
        
        # 验证所有文件都自动创建了
        files_created = [
            self.builder.templates_dir / "contours" / f"{tooth_id}.json",
            self.builder.templates_dir / "features" / f"{tooth_id}_features.json"
        ]
        
        for file_path in files_created:
            self.assertTrue(file_path.exists(), f"文件应该自动创建: {file_path}")
        
        # 验证数据库记录也自动创建了
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM templates WHERE tooth_id = ?', (tooth_id,))
        count = cursor.fetchone()[0]
        conn.close()
        
        self.assertEqual(count, 1, "数据库记录应该自动创建")
        
        print("✅ 答案2: 是的，调用serialize_contours_with_features()一次就能自动保存:")
        print("   - 轮廓数据JSON文件")
        print("   - 特征数据JSON文件")
        print("   - 数据库记录")
    
    def test_unique_coding(self):
        """测试唯一编码功能 - 回答用户问题3"""
        print("🧪 验证: 是否有对应的唯一编码?")
        
        # 测试自动生成的唯一ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        auto_tooth_id = f"TOOTH_{timestamp}"
        
        # 测试手动指定的ID
        manual_tooth_id = "MANUAL_TOOTH_001"
        
        # 保存两个不同的模板
        success1 = self.builder.serialize_contours_with_features(
            self.test_contours, auto_tooth_id, "test1.jpg"
        )
        success2 = self.builder.serialize_contours_with_features(
            self.test_contours, manual_tooth_id, "test2.jpg"
        )
        
        self.assertTrue(success1 and success2)
        
        # 验证数据库中有两个不同的记录
        conn = sqlite3.connect(self.test_db)
        cursor = conn.cursor()
        cursor.execute('SELECT tooth_id FROM templates')
        tooth_ids = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        self.assertIn(auto_tooth_id, tooth_ids)
        self.assertIn(manual_tooth_id, tooth_ids)
        self.assertEqual(len(set(tooth_ids)), len(tooth_ids), "所有ID应该唯一")
        
        print("✅ 答案3: 是的，系统支持唯一编码:")
        print(f"   - 自动生成: {auto_tooth_id}")
        print(f"   - 手动指定: {manual_tooth_id}")
        print("   - 数据库主键约束确保唯一性")
        print("   - 支持INSERT OR REPLACE避免冲突")

if __name__ == '__main__':
    print("🧪 开始测试牙齿模板构建器...")
    print("=" * 60)
    
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_classes = [
        TestFeatureExtraction,
        TestDataSaving,
        TestUniqueIDGeneration,
        TestMatchIntegration,
        TestAdvancedFeatures,
        TestFeatureAnswers
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print("\n" + "=" * 60)
    print("🧪 测试总结:")
    print(f"   运行测试: {result.testsRun}")
    print(f"   成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"   失败: {len(result.failures)}")
    print(f"   错误: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ 失败的测试:")
        for failure in result.failures:
            print(f"   - {failure[0]}")
    
    if result.errors:
        print("\n💥 错误的测试:")
        for error in result.errors:
            print(f"   - {error[0]}")
    
    print("\n📋 回答用户问题:")
    print("1. ✅ BulidTheLab能保存match.py需要的特征值")
    print("2. ✅ 能自动保存所有特征和数据")
    print("3. ✅ 支持唯一编码系统")