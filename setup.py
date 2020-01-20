from setuptools import setup

setup(name='drl_playground',
      version='0.1',
      description='Experiments for me to get more familiar with DRL work.',
      url='https://github.com/yutingyue514/drl_playground',
      author='Summer Yue',
      author_email='yutingyue514@gmail.com',
      license='MIT',
      packages=['drl_playground'],
      install_requires=['absl-py', 'gym', 'tensorflow', 'tqdm'],
      zip_safe=False)
