from setuptools import setup, find_packages

setup(name='madrigal',
      version='0.1',
      description='Multimodal learning for drug combination outcome prediction',
      packages=find_packages(include=['madrigal', 'madrigal.*']),
      author='zitnik-lab',
      zip_safe=False,
      python_requires='>=3.8')
