import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IndexedConv",
    version="0.1.0a0",
    author="M. Jacquemont",
    author_email="jacquemont@lapp.in2p3.fr",
    description="An implementation of indexed convolution and pooling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IndexedConv/IndexedConv",
    install_requires=[
        "torch>=0.4",
        "numpy",
        "tensorboardx",
        "matplotlib",

    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
