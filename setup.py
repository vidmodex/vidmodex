import setuptools
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vidmodex",
    version="1.0rc",
    author="anonymous",
    author_email="anonymous@gmail.com",
    description="Code for Model Extration for Video and Image Classification",
    long_description=long_description,
    packages=["vidmodex"],
    install_requires=['timm','tqdm','moviepy', 'fvcore','tensorboard', 'pandas', 'dotmap', 'torchvideo2', 'einops', f'shap @ file://{base_dir}/external_src/shap']
)
