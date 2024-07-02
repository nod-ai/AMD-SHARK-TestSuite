from setuptools import find_namespace_packages, setup


#### SHARK TEST SUITE SETUP ####

setup(
    name=f"ireers",
    version="1.0.0",
    author="SHARK Authors",
    author_email="esaimana@amd.com",
    description="SHARK Test Suite Tools",
    url="https://github.com/nod-ai/SHARK-TestSuite",
    packages=find_namespace_packages(
        include=[
            "ireers",
            "ireers.*",
        ],
    ),
)
