# SSMuLA

## About

Code base for Site Saturation Mutagenesis Landscape Analysis

### Datasets
* The `data` folder is structured based on the landscape name, for each:
    - `fasta`: the fasta file for the parent
    - `fitness_landscape`: the csv file for the mutants and fitness values
    - `processed`: processed fitness file
#### Three-site
* DHFR
    * A dihydrofolate reductasae (DHFR) encoded by *E. coli folA* gene [Papkou *et al.*, Science (2023)](https://www.science.org/doi/full/10.1126/science.adh3860)
    * 9 mucleotides at 3 amino aicd positions (A26-D27-L28)
    * Almost 4^9 (262,144) DNA genotypes
    * Map fitness to trimethoprim resistance
    * Sixfold replicates
* TrpB
    * LibA
    * LibB
    * LibC
    * LibD
    * LibE
    * LibF
    * LibG
    * LibH
#### Four-site
* GB1
* TrpB

### Zero-shot
* 

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
