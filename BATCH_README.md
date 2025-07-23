# 🦷 批量牙齿图像处理器使用指南

## 📋 功能概述

现在你的 `BulidTheLab.py` 已经支持批量处理功能！主要特性：

✅ **自动批量处理** - 一次处理整个目录的图像  
✅ **智能颜色复用** - 第一张图选色，后续自动应用  
✅ **进度监控** - 实时显示处理进度和统计信息  
✅ **错误处理** - 自动跳过问题文件，生成详细报告  
✅ **去重处理** - 自动跳过已处理的图像（可选）  
✅ **连续编号** - 自动生成 TOOTH_001, TOOTH_002... 

## 🚀 快速开始

### 1. 准备图像文件
将要处理的牙齿图像放入 `images/` 目录：
```
images/
├── Tooth_1.png
├── Tooth_2.png  
├── Tooth_3.png
└── ...
```

### 2. 运行批量处理
```bash
# 基本批量处理
python BulidTheLab.py --batch

# 指定输入目录
python BulidTheLab.py --batch --input-dir "D:/my_tooth_images"

# 指定输出目录  
python BulidTheLab.py --batch --input-dir "images" --output-dir "my_templates"
```

### 3. 处理流程
1. 🔍 **扫描文件** - 自动发现所有图像文件
2. 🎨 **选择颜色** - 在第一张图上点击选择目标颜色
3. ⚡ **批量处理** - 自动处理所有图像
4. 📊 **生成报告** - 显示处理统计和结果

## 📚 详细用法

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--batch` | 启用批量处理模式 | - |
| `--input-dir` | 输入图像目录 | `images` |  
| `--output-dir` | 输出模板目录 | `templates` |
| `--database` | 数据库文件路径 | `tooth_templates.db` |
| `--skip-processed` | 跳过已处理的文件 | `True` |
| `--single-image` | 处理单张图像 | - |

### 使用示例

```bash
# 1. 基本批量处理（推荐）
python BulidTheLab.py --batch

# 2. 处理特定目录
python BulidTheLab.py --batch --input-dir "D:/tooth_photos"

# 3. 重新处理所有文件（包括已处理的）
python BulidTheLab.py --batch --skip-processed False

# 4. 单张图像处理
python BulidTheLab.py --single-image "path/to/tooth.jpg"

# 5. 传统模式（使用PHOTO_PATH）
python BulidTheLab.py
```

## 🔧 编程接口

你也可以在Python代码中直接使用：

```python
from BulidTheLab import BatchToothProcessor

# 创建批量处理器
processor = BatchToothProcessor(
    input_dir="images",
    templates_dir="templates", 
    database_path="tooth_templates.db"
)

# 批量处理
report = processor.process_batch(
    skip_processed=True,     # 跳过已处理文件
    interactive_first=True,  # 第一张图交互选色
    show_progress=True       # 显示进度
)

print(f"✅ 成功处理: {report['processed']} 个文件")
```

## 📊 处理报告

批量处理完成后会显示详细报告：

```
============================================================
🎉 批量处理完成！
============================================================
📊 处理统计:
   🔍 发现文件: 6 个
   ✅ 成功处理: 5 个  
   ❌ 处理失败: 1 个
   ⏭️  跳过文件: 0 个
   📈 成功率: 83.3%

❌ 失败文件详情:
   • corrupted_image.png: 未检测到有效轮廓
```

## 🎯 支持的图像格式

- PNG (*.png)
- JPEG (*.jpg, *.jpeg)
- BMP (*.bmp) 
- TIFF (*.tiff, *.tif)

## 📁 输出结构

批量处理后会生成：

```
templates/
├── contours/           # JSON轮廓数据
│   ├── TOOTH_001.json
│   ├── TOOTH_002.json
│   └── ...
├── features/          # 特征数据
│   ├── TOOTH_001_features.json
│   └── ...
└── images/           # 轮廓可视化图像
    ├── TOOTH_001.png
    └── ...

tooth_templates.db    # SQLite数据库
```

## ❓ 常见问题

**Q: 如何修改颜色检测参数？**  
A: 在第一张图选色时，选择多个代表性点，系统会自动计算最佳参数。

**Q: 处理失败的图像怎么办？**  
A: 检查报告中的失败原因，通常是图像质量或轮廓检测问题，可以手动调整图像后重新处理。

**Q: 如何重新处理某些图像？**  
A: 使用 `--skip-processed False` 参数，或者删除数据库中对应记录。

**Q: 可以中断后继续处理吗？**  
A: 是的，系统会自动跳过已处理的文件，支持断点续传。

## 🔄 下一步计划

- [ ] 并行处理支持（多进程）
- [ ] 自动颜色检测（无需手动选择）
- [ ] 批量处理配置文件
- [ ] Web界面支持
- [ ] 处理质量评估

---

需要帮助？请检查控制台输出的详细信息或联系开发者。
