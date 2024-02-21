# SSMuLA

## About

Code base for Site Saturation Mutagenesis Landscape Analysis

### Datasets
* The `data` folder is structured based on the landscape name, for each:
    - `fasta`: the fasta file for the parent
    - `fitness_landscape`: the csv file for the mutants and fitness values
    - `processed`: processed fitness file

#### Three systems
##### DHFR
* A dihydrofolate reductasae (DHFR) encoded by *E. coli folA* gene [Papkou *et al.*, Science (2023)](https://www.science.org/doi/full/10.1126/science.adh3860)
* 9 mucleotides at 3 amino aicd positions (A26, D27, L28)
* Almost 4^9 (262,144) DNA genotypes
* Map fitness to trimethoprim resistance
* Sixfold replicates
* PDB ID: 6XG5
* Define active / functional mutations with a mixture Gaussian model to the fitness values using the `mixEM` function from the `mixtools` in R package, by setting the number of components (distributions) to k=3 and using the bell-shaped distribution’s 0.999 quantile as fitness cut-off value
##### TrpB
* A thermostable β-subunit of tryptophan synthase [Johnston *et al.*, PNAS (2024)]
* Synthesizes L-tryptophan from indol and L-serine
* Select for ability to complement Trp auxotrophy supplied with exogenous indole
* Connects TrpB activity to growth rate
* Hyperthermophile *Thermotoga maritima*, Tm9D8* was selected as the parent enzyme
* 4-site-saturation landscape (V183, F184, V227, S228)
* Fitness calcualted from multiple timepoints and aggregared into a final fitness score per vairant
* Two replicates
* PDB ID: 8VHH
* Define active as 1.96 standard deviations above the mean fitness of all stop-codon-containing sequences over both replicates
* LibA
* LibB
* LibC
* LibD
* LibE
* LibF
* LibG
* LibH
* Four-site
##### GB1
* PDB ID: 2GI9

### Preprocessing
* Processed: Rename columns to be `AAs`, `AA1`, `AA2`, `AA3`, `AA4`, `fitness`, `active`
* 

### Zero-shot
#### Triad
#### ESM
#### EVMutation
* All EVMutation predictions run with [EVcouplings](https://v2.evcouplings.org/)
* All settings remain default
* Model parameters are downloaded and renamed
* Use the recommneded results and full results can be obtained from the links

- [GB1](https://v2.evcouplings.org/results/c12744e78f4744ee8102d8021c243398)
- [TrpB](https://v2.evcouplings.org/results/db0701748dce4dfe8f27ac5a535391bf)


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
