from setuptools import setup, find_packages

# Read long description from README.md
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except:
    long_description = "EDAwala - AI-Powered Exploratory Data Analysis Tool"

setup(
    name="edawala",
    version="0.1.0",
    author="Yashu",
    author_email="kumaryashu496@gmail.com",
    description="Advanced EDA toolkit with AI-powered analysis features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/syashu16/edawala",
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "scikit-learn>=1.0.0",
        "streamlit>=1.10.0",
        "jinja2>=3.0.0",
        "plotly>=5.0.0",
        "scipy>=1.7.0",
        "google-generativeai>=0.1.0",
    ],
    extras_require={
    "pdf": ["weasyprint>=52.5", "pdfkit>=1.0.0", "reportlab>=3.6.1"],
    "notebook": ["nbformat>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "edawala=edawala.cli:main",
        ],
    },
)