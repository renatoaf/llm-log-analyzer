from setuptools import setup, find_packages
import pathlib

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="llm-log-analyzer",  # pip install name (can contain -)
    version="1.0.0",
    description="AI-powered log analyzer for CI environments using LLMs",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/renatoaf/llm-log-analyzer",
    license="MIT",
    keywords="unity build log analyzer ci automation ai llm root cause analysis",
    package_dir={"": "src"},
    package_data={
        "llm_log_analyzer": [
            "prompts/*.txt",
            "patterns/*.txt",
        ],
    },
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "anthropic>=0.34.0",    
        "google-generativeai>=0.3.0",
        "openai>=1.0.0",
        "boto3>=1.28.0",
        "click>=8.1.0",
        "requests>=2.31.0",
        "tiktoken>=0.5.0",
        "langchain-text-splitters>=0.2.0"
    ],
    entry_points={
        "console_scripts": [
            "llm-log-analyzer=llm_log_analyzer.analyze_log:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
