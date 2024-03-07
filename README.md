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
* A dihydrofolate reductase (DHFR) encoded by *E. coli folA* gene [Papkou *et al.*, Science (2023)](https://www.science.org/doi/full/10.1126/science.adh3860)
* 9 nucleotides at 3 amino acid positions (A26, D27, L28)
* Almost 4^9 (262,144) DNA genotypes
* Map fitness to trimethoprim resistance
* Sixfold replicates
* PDB ID: 6XG5
* Define active / functional mutations with a mixture Gaussian model to the fitness values using the `mixEM` function from the `mixtools` in R package, by setting the number of components (distributions) to k=3 and using the bell-shaped distribution’s 0.999 quantile as fitness cut-off value
##### TrpB
* A thermostable β-subunit of tryptophan synthase [Johnston *et al.*, PNAS (2024)]
* Synthesizes L-tryptophan from indole and L-serine
* Select for ability to complement Trp auxotrophy supplied with exogenous indole
* Connects TrpB activity to growth rate
* Hyperthermophile *Thermotoga maritima*, Tm9D8* was selected as the parent enzyme
* 4-site-saturation landscape (V183, F184, V227, S228)
* Fitness calculated from multiple timepoints and aggregated into a final fitness score per variant
* Two replicates
* PDB ID: 8VHH
* Define active as 1.96 standard deviations above the mean fitness of all stop-codon-containing sequences over both replicates
* Nine 3-site libraries and one 4-site library
    * TrpB3A: A104, E105, T106
    * TrpB3B: E105, T106, G107
    * TrpB3C: T106, G107, A108
    * TrpB3D: T117, A118, A119
    * TrpB3E: F184, G185, S186
    * TrpB3F: L162, I166, Y301
    * TrpB3G: V227, S228, Y301
    * TrpB3H: S228, G230, S231
    * TrpB3I: Y182, V183, F184
    * TrpB4: V183, F184, 227V, S228
##### GB1
* PDB ID: 2GI9
* 4-site-saturation landscape (V39, D40, G41, V54)

### Preprocessing
* Processed with `fitness_process_vis`
* Rename columns to be `AAs`, `AA1`, `AA2`, `AA3`, `AA4`, `fitness`, add `active` if not already there and add `muts` columns
* Scale to parents or max

### Simulations
#### DE
* Run `de_simulations` and visualise with `plot_de_simulations`
#### MLDE

### Zero-shot
* Note that esm and evmutation scores are generated with corresponeding scrips from `p2g4s2m`
#### Triad
* Prep mutation file in `.mut` format such as `A_1A+A_2A+A_3A+A_4A` with `TriadGenMutFile` class in `SSMuLA.triad_prepost` 
* With `triad-2.1.3` local command line
* Prepare structure with `2prep_structures.sh`
* Ran `3getfixed.sh`
* Parse results with `ParseTriadResults` class in `SSMuLA.triad_prepost` 
#### ESM
* 
#### EVMutation
* All EVMutation predictions run with [EVcouplings](https://v2.evcouplings.org/)
* All settings remain default
* Model parameters in the `.model` files are downloaded and renamed
* Use the recommneded results and full results can be obtained from the links
- [DHFR](https://v2.evcouplings.org/results/057bbf069e4b43789bd87bd22b9982ab)
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
