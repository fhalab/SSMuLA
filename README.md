# SSMuLA

## About

Code base for Site Saturation Mutagenesis Landscape Analysis

## Prepare for installation

If you don't already have twine installed run the following:
`python -m pip install --user --upgrade twine`

## The following will create the package
```
python setup.py sdist bdist_wheel
twine check dist/SSMuLA-1.0.0.tar.gz
```

## Install to python environment localling

`pip install PATH_TO_SSMuLA/dist/SSMuLA-1.0.0.tar.gz`
You should run this before uploading it and check all works as expected.

## The following will push the package to pip 
**Note you need to set up a pip account first**

```
twine upload dist/*
```

## Have a look at your projects page on pip

`https://pypi.org/project/SSMuLA/`
