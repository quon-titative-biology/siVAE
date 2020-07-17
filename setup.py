from setuptools import setup, find_packages

setup(name='siVAE',
      version='0.1',
      description='scalable and interpretable Variational Autoencoder',
      url='',
      author='Yongin Choi, Gerald Quon',
      author_email='yonchoi@ucdavis.edu',
      license='MIT',
      packages=find_packages(),
      install_requires=[
            'pandas',
            'matplotlib',
            'scikit-learn',
            'seaborn',
            'tensorflow==1.15',
            "tensorflow-probability==0.8.0",
            'scipy',
            'scikit-image',
            'scanpy',
            'gseapy'
      ],
      zip_safe=False
      )
