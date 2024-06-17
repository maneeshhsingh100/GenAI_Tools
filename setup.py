from setuptools import setup, find_packages

setup(
    name='speech_model_lib',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'transformers',
        'torch',
        'librosa',
        'torchaudio',
        'numpy',
        'matplotlib',
    ],
    author='Manish Singh',
    description='An end-to-end speech model library using pretrained models',
    url='https://github.com/maneeshhsingh100/GenAI_Tools/speech_model_lib',
)
