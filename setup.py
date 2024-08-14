from setuptools import find_packages,setup
def get_requirements(filepath):
    requirements=[]
    with open(filepath) as file_obj:
        requirements=file_obj.readlines()
        for i in requirements:
            requirements=[i.replace('/n','')]
        if '-e .' in requirements:
            requirements.remove('-e .')
    print(requirements)
    return requirements




setup(name='Mercedes Benz Greener Manufacturing Using Regression Techniques',
      version='0.0.1',
      author='Aswith Sama',
      author_email='asama@hawk.iit.edu',
      packages=find_packages(),
      install_requires=get_requirements('requirements.txt')
)


