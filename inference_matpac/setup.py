from pathlib import Path
from setuptools import setup, find_packages


def get_readme_text():
  root_dir = Path(__file__).parent
  readme_path = root_dir / "README.md"
  return readme_path.read_text()


setup(
    name="matpac",
    version="0.0.1",
    author="Aurian Quelennec",
    url='https://github.com/aurianworld/matpac',
    description='Effective SSL fundation model for general audio',
    long_description=get_readme_text(),
    long_description_content_type='text/markdown',
    install_requires=["numpy==1.26.3",
                      "einops==0.7.0",
                      "timm==0.4.12",
                      "torch==2.4.1",
                      "torchaudio==2.4.1",
                      "torchvision==0.19.1"],
    python_requires='>=3.9.18',
    packages=find_packages(include=["matpac*"])
)
