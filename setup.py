import codecs
from setuptools import setup, find_packages

with codecs.open('README.md', 'r', 'utf8') as reader:
    long_description = reader.read()

with codecs.open('requirements.txt', 'r', 'utf8') as reader:
    install_requires = list(map(lambda x: x.strip(), reader.readlines()))
    
try:
    import cupy
except:
    print('CuPy is not available. Please install it manually: https://docs-cupy.chainer.org/en/stable/install.html#install-cupy')
#     print('Cupy not installed. Package will install cupy, which will take several minutes')
#     print('If you would like to install Cupy yourself, check here https://docs-cupy.chainer.org/en/stable/install.html#install-cupy')
#     install_requires.append('cupy')

setup(
    name='SpeedTorch',
    version='0.1.1',
    packages=find_packages(),
    url='https://github.com/Santosh-Gupta/SpeedTorch',
    license='MIT',
    author='Santosh Gupta',
    author_email='SanGupta.ML@gmail.com',
    description='Fast Pinned CPU -> GPU transfer',
    long_description='Fast Pinned CPU -> GPU transfer',
    long_description_content_taype='text/markdown',
    python_requires='>=3.5.0',
    install_requires=install_requires,
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
