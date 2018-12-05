import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IndexedConv",
    version="0.1.0a",
    author="M. Jacquemont, T. Vuillaume, L. Antiga",
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
        "h5py",
        "sphinxcontrib-katex"

    ],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    license='MIT',
)
