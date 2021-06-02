from setuptools import setup
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='iar',
      version = '1.1.0',
      description='Irregularly Observed Autoregressive Models',
      url='https://github.com/felipeelorrieta/iAR',
      download_url="https://github.com/felipeelorrieta/iAR/archive/refs/tags/v1.0.0.tar.gz",
      author='Felipe Elorrieta',
      author_email='felipe.elorrieta@usach.cl',
      license='MIT',
      packages=['iar'],
      keywords = ['irregulary observed time series','autoregressive'],
      install_requires=[
        'numpy','pandas','scipy','matplotlib','sklearn','statsmodels'
    ],
      long_description=long_description,
      long_description_content_type='text/markdown',
      include_package_date=True,
      package_dir={'iar':'iar'},
      package_data={"":["../data/*.csv"]},
      zip_safe=False)
