import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="oscsipm",
    version="0.4.1",
    author="Pavel Gostev",
    author_email="gostev.pavel@physics.msu.ru",
    description="Instruments to make photocounting statistics from histograms and raw oscillograms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vongostev/OscSiPM",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    include_package_data=True,
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'lecroyparser',
        'joblib',
        'dataclasses',
        'compress_pickle',
        'tekwfm2',
        'fpdet'
    ],
)
