#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理牙齿图像示例脚本
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from BulidTheLab import BatchToothProcessor

def quick_batch_test():
    """快速批量处理测试"""
    print("🚀 快速批量处理测试")
    print("=" * 50)
    
    # 检查images目录是否存在
    images_dir = Path("images")
    if not images_dir.exists():
        print(f"❌ images目录不存在，请创建并放入图像文件")
        return
    
    # 创建批量处理器
    processor = BatchToothProcessor(
        input_dir="images",
        templates_dir="templates", 
        database_path="tooth_templates.db"
    )
    
    try:
        # 开始批量处理
        report = processor.process_batch(
            skip_processed=True,     # 跳过已处理的文件
            interactive_first=True,  # 第一张图交互选色
            show_progress=True       # 显示进度
        )
        
        print(f"\n🎉 测试完成!")
        print(f"✅ 成功处理: {report['processed']} 个文件")
        print(f"❌ 处理失败: {report['failed']} 个文件")
        print(f"⏭️  跳过文件: {report['skipped']} 个文件")
        
        return report
        
    except Exception as e:
        print(f"❌ 批量处理失败: {e}")
        return None

def demonstrate_usage():
    """演示不同的使用方法"""
    print("📚 批量处理器使用演示")
    print("=" * 50)
    
    # 方法1: 基本用法
    print("\n1️⃣ 基本批量处理:")
    print("   python BulidTheLab.py --batch")
    
    # 方法2: 指定目录
    print("\n2️⃣ 指定输入输出目录:")
    print("   python BulidTheLab.py --batch --input-dir D:/tooth_images --output-dir D:/templates")
    
    # 方法3: 处理单张图片
    print("\n3️⃣ 处理单张图片:")
    print("   python BulidTheLab.py --single-image path/to/image.jpg")
    
    # 方法4: 不跳过已处理文件
    print("\n4️⃣ 重新处理所有文件:")
    print("   python BulidTheLab.py --batch --no-skip-processed")
    
    print("\n📝 使用说明:")
    print("   • 批量处理会自动扫描指定目录下的图像文件")
    print("   • 第一张图片需要手动选择颜色，后续自动应用")
    print("   • 支持 PNG, JPG, JPEG, BMP, TIFF 格式")
    print("   • 自动生成连续编号: TOOTH_001, TOOTH_002...")
    print("   • 自动跳过已处理的文件（可通过参数控制）")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        demonstrate_usage()
    else:
        quick_batch_test()
