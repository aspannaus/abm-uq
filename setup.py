from setuptools import setup, find_packages

setup(
    name='abm-uq',
    version='0.0.1',
    description='',
    url='https://github.com/aspannaus/abm-uq',
    author='Adam Spannaus',
    author_email='spannausat@ornl.gov',
    license='MIT',
    install_requires=['jax',
                      'matplotlib',
                      'scikit-learn',
                      'pyyaml',
                      'pandas',
                      'numpy',
                      'mpi4py',
                      'networkx',
                      'repast4py',
                      'scipy'
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.11',
    ],
)