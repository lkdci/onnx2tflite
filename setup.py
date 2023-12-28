from setuptools import find_packages
from setuptools import setup


def read_readme(readme_path: str, decode: bool = False):
    if decode:
        with open(readme_path, "rb") as f:
            return f.read().decode("utf-8")
    else:
        with open(readme_path) as f:
            return f.read()


def read_requirements(req_path: str):
    with open(req_path) as f:
        requirements = f.read().splitlines()
    return [r for r in requirements if r.strip() and not r.lstrip().startswith("#")]


setup(
    name="onnx2tflite",
    version="0.0.2",
    description="Tool for onnx->keras or onnx->tflite",
    author="MPolaris",
    url="https://github.com/MPolaris/onnx2tflite",
    keywords=["onnx", "tflite", "keras", "conversion"],
    include_package_data=True,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.6.9",
    install_requires=read_requirements("requirements.txt"),
    long_description=read_readme("readme.md"),
    long_description_content_type="text/markdown",
    entry_points={"console_scripts": ["converter = onnx2tflite.__main__:run"]},
    license="Apache License v2.0",
)
