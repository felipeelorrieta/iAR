from setuptools import setup

setup(name='iar',
      version = '1.0.0',
      description='Irregularly Observed Autoregressive Models',
      url='https://github.com/felipeelorrieta',
      author='Felipe Elorrieta',
      author_email='felipe.elorrieta@usach.cl',
      license='MIT',
      packages=['iar'],
      keywords = ['irregulary observed time series','autoregressive'],
      install_requires=[
        'numpy','pandas','scipy','matplotlib','sklearn'
    ],
      include_package_date=True,
      package_data={"":["data/*.csv"]},
      zip_safe=False)
