# coding=utf-8

from setuptools import setup, find_packages

setup(
    # 基础元数据
    name="food",  # 包名称（在 PyPI 上唯一）
    version="0.1.0",           # 版本号（遵循语义化版本规范）
    author="ll",
    author_email="your.email@example.com",
    description="I will describe it later",
    long_description=open("readme.md",encoding="utf-8").read(),  # 长描述（通常从 README 读取）
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your_repository",  # 项目地址
    
    # 包内容配置
    packages=find_packages(),  # 自动发现所有包（包含 __init__.py 的目录）
    # 或手动指定包：
    # packages=["your_package", "your_package.submodule"],
    
    # 依赖项
    install_requires=[        # 必需依赖项列表
        "boxsers == 1.5.2",
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scipy', 'scikit-learn', 'tensorflow',
                      'tables', 'scikit-image'
    ],
    extras_require={          # 可选依赖项
    },
    
    # 包含非代码文件（如数据、配置文件）
    include_package_data=True,  # 需配合 MANIFEST.in 或 package_data 使用
    package_data={
        "your_package": ["data/*.csv", "configs/*.yaml"],
    },
    
    # 命令行工具（将函数映射为终端命令）
    entry_points={
        "console_scripts": [
            "your-command = your_package.cli:main",  # 例如：cli.py 中的 main 函数
        ],
    },
    
    # 分类信息（可选）
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",  # Python 版本要求
)


# if __name__ == "__main__":
#     setup()