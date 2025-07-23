#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理使用示例集合
"""

def show_usage_examples():
    """显示各种使用方式的示例"""
    print("🦷 批量牙齿图像处理器 - 使用示例")
    print("=" * 60)
    
    print("\n📚 1. 基本批量处理")
    print("   命令: python BulidTheLab.py --batch")
    print("   说明: 处理 images/ 目录下的所有图像")
    print("   流程: 扫描文件 → 第一张图选色 → 批量自动处理")
    
    print("\n📚 2. 指定输入目录")
    print("   命令: python BulidTheLab.py --batch --input-dir 'D:/tooth_photos'")
    print("   说明: 处理指定目录下的图像文件")
    
    print("\n📚 3. 指定输出目录")  
    print("   命令: python BulidTheLab.py --batch --output-dir 'my_templates'")
    print("   说明: 将模板保存到指定目录")
    
    print("\n📚 4. 完整自定义路径")
    print("   命令: python BulidTheLab.py --batch \\")
    print("           --input-dir 'D:/photos' \\")
    print("           --output-dir 'D:/results' \\")
    print("           --database 'my_teeth.db'")
    
    print("\n📚 5. 重新处理所有文件")
    print("   命令: python BulidTheLab.py --batch --skip-processed False")
    print("   说明: 包括已处理的文件重新处理")
    
    print("\n📚 6. 单张图像处理")
    print("   命令: python BulidTheLab.py --single-image 'path/to/tooth.jpg'")
    print("   说明: 只处理指定的单张图像")
    
    print("\n📚 7. 传统交互模式")
    print("   命令: python BulidTheLab.py")
    print("   说明: 使用PHOTO_PATH处理单张图像（需要交互）")

def show_workflow():
    """显示批量处理工作流程"""
    print("\n🔄 批量处理工作流程")
    print("=" * 60)
    
    steps = [
        "📁 扫描输入目录，发现所有图像文件",
        "🔍 检查数据库，跳过已处理的文件（可选）", 
        "🎨 第一张图像：手动点击选择目标颜色",
        "⚙️  计算HSV颜色参数和阈值范围",
        "🚀 后续图像：自动应用颜色参数处理",
        "🔬 对每张图像执行：",
        "   • HSV颜色掩码生成",
        "   • 形态学分离黏连区域", 
        "   • 轮廓检测和特征提取",
        "   • 自动生成TOOTH_XXX编号",
        "   • 保存到数据库和文件系统",
        "📊 生成详细的批量处理报告"
    ]
    
    for i, step in enumerate(steps, 1):
        if step.startswith("   "):
            print(f"     {step[3:]}")
        else:
            print(f"{i:2d}. {step}")

def show_output_structure():
    """显示输出文件结构"""
    print("\n📁 输出文件结构")
    print("=" * 60)
    
    structure = """
工作目录/
├── images/                     # 输入图像目录
│   ├── Tooth_1.png
│   ├── Tooth_2.png  
│   └── ...
├── templates/                  # 输出模板目录
│   ├── contours/              # JSON轮廓数据
│   │   ├── TOOTH_001.json     # 完整轮廓和特征数据
│   │   ├── TOOTH_002.json
│   │   └── ...
│   ├── features/              # 纯特征数据
│   │   ├── TOOTH_001_features.json
│   │   └── ...
│   └── images/                # 轮廓可视化图像
│       ├── TOOTH_001.png      # 带轮廓标记的图像
│       └── ...
├── tooth_templates.db         # SQLite数据库
├── BulidTheLab.py            # 主程序
└── batch_example.py          # 批量处理示例
"""
    print(structure)

def show_benefits():
    """显示批量处理的优势"""
    print("\n✨ 批量处理优势")
    print("=" * 60)
    
    benefits = [
        "⚡ 高效率：一次处理多个文件，无需逐个操作",
        "🎯 一致性：统一的颜色参数确保处理结果一致",
        "🛡️ 稳定性：自动错误处理，跳过问题文件继续运行", 
        "📊 可追踪：详细的进度显示和处理报告",
        "🔄 可恢复：自动跳过已处理文件，支持断点续传",
        "🎛️ 可配置：支持多种命令行参数自定义处理方式",
        "📈 可扩展：基于模块化设计，易于添加新功能"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")

def show_tips():
    """显示使用技巧"""
    print("\n💡 使用技巧")
    print("=" * 60)
    
    tips = [
        "🎨 颜色选择：在第一张图像上选择多个代表性点获得更好效果",
        "📁 文件组织：将相似类型的牙齿图像放在同一目录中",
        "🔍 质量控制：处理前检查图像质量，去除模糊或损坏的文件",
        "💾 数据备份：定期备份数据库和模板文件",
        "📊 监控日志：注意控制台输出的错误信息和警告",
        "⚙️ 参数调优：可以修改代码中的HSV阈值范围来适应不同图像",
        "🔄 重新处理：如果结果不满意，可以删除数据库记录重新处理"
    ]
    
    for tip in tips:
        print(f"  {tip}")

if __name__ == "__main__":
    show_usage_examples()
    show_workflow()
    show_output_structure()
    show_benefits()
    show_tips()
    
    print(f"\n🎉 准备开始批量处理了吗？")
    print(f"💡 运行: python BulidTheLab.py --batch")
    print(f"💡 或者: python batch_example.py")
