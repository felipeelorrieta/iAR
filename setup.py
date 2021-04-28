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
        'numpy >=1.9.0',
    ],
      zip_safe=False)
