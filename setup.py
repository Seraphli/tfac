from setuptools import setup, find_packages
import re


def get_property(prop, project):
    result = re.search(r'{}\s*=\s*[\'"]([^\'"]*)[\'"]'.format(prop),
                       open(project + '/__init__.py').read())
    return result.group(1)


setup(
    name="tfac",
    version=get_property("__version__", "tfac"),
    description="package for accelerate tensorflow input pipeline",
    author="SErAphLi",
    url="https://github.com/Seraphli/tfac.git",
    packages=find_packages()
)
