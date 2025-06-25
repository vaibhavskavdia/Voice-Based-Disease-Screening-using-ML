from setuptools import find_packages,setup
from typing import List
HYPHEN_E_DOT="-e ."
def get_requirements(file_path:str)-> List[str]:
    '''this function will return the list of requirements'''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj
        requirements=[req.replace("\n","") for req in requirements]
        
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
            #now whenever i call pip install requirement.txt it will directly initialize the setup.py file 
            #with the use of -e . command and this if statement removes it to ignore it as a library
    return requirements

setup(
    name="Voice-Based-Disease-Screening-using-ML",
    version="0.0.1",
    author="Vaibhav Kavdia",
    author_email="vaibhavskavdia@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirement.txt')
)