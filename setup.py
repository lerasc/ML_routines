from setuptools import setup,find_packages

setup(
    name            =  'ML_routines',
    version         =  '1.0.0',
    packages        = ['ML_routines'], # can also write: find_packages()
    url             =  '',
    license         =  '',
    author          ='Sandro C. Lera',
    author_email    ='sandrolera@gmail.com',
    description     ='machine learning routines',
    python_requires ='>3.5.2',
    install_requires=[
                        "numpy>=1.20.3",
                        "pandas>=1.3.4",
                        "seaborn>=0.11.2",
                        "scikit-learn>=1.0.1",
                    ]
)
