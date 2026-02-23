from setuptools import find_packages, setup
from typing import List

HYPHEN_DOT_E = "-e ."
def get_requirements(file_path:str)->List[str]:
    #this is the fn to retrive all package name from requirements.txt and return it as a list
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [line.replace("\n", "") for line in requirements]

        if HYPHEN_DOT_E in requirements:
            requirements.remove(HYPHEN_DOT_E)
    return requirements


setup(
    name="student-performance-prediction-project",
    version='0.0.1',
    author="Shibith",
    author_email="shibithp94@gmail.com",
    packages=find_packages(),
    install_requires= get_requirements('requirements.txt') #['pandas', 'numpy', 'seaborn'] -> but auto retrived from requiremnts.txt through the function
)

#"We can directly install this setup.py or trigger to install this file when installing requirements.txt by adding '-e .'"