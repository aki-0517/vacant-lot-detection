from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vacant-lot-detection",
    version="0.1.0",
    author="Land Vacancy Detection Team",
    author_email="team@example.com",
    description="A computer vision system for detecting vacant lots in aerial and satellite imagery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/vacant-lot-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "opencv-python",
        "pyyaml",
        "numpy",
        "Pillow",
        "tqdm",
        "scikit-learn",
        "tensorboard",
        "matplotlib",
        "seaborn",
        "albumentations",
        "segmentation-models-pytorch",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
            "isort",
            "pre-commit",
        ],
        "notebooks": [
            "jupyter",
            "ipywidgets",
            "plotly",
        ],
    },
    entry_points={
        "console_scripts": [
            "vacant-lot-preprocess=data_pipeline.preprocess:main",
            "vacant-lot-train=train:main",
            "vacant-lot-evaluate=evaluate:main",
            "vacant-lot-inference=inference:main",
        ],
    },
)