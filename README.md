# SSMuLA

## About

* Code base termed "Site Saturation Mutagenesis Landscape Analysis (SSMuLA)" for paper titled "Evaluation of Machine Learning-Assisted Directed Evolution Across Diverse Combinatorial Landscapes"
* Data and results can be found at [Zenodo](10.5281/zenodo.13910506)

### Datasets
* The `data` folder is structured based on the landscape name, for each:
    - `.fasta`: the fasta file for the parent
    - `.pdb`: the pdb file for the parent
    - `.model`: the EVmutation model file
    - `fitness_landscape`: the folder containing csv files for the mutants and fitness values from the original sources
    - `scale2max`: the folder containing processed fitness csv files returned from the `process_all` function in the `SSMuLA.fitness_process_vis` module

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
* Run
```
python -m tests.test_preprocess
```
* Processed with `fitness_process_vis`
* Rename columns to be `AAs`, `AA1`, `AA2`, `AA3`, `AA4`, `fitness`, add `active` if not already there and add `muts` columns
* Scale to `max` (with option to scale to `parent`)
* Processed data saved in `scale2max` folder
* The landcape stats will be saved 


### Landscape attributes
#### Local optima
* Run
```
python -m tests.local_optima
```
* Calculate local optima with `calc_local_optima` function in `SSMuLA.local_optima`

#### Pairwise epistasis
* Run
```
python -m tests.pairwise_epistasis
```
* Calculate pairwise epistasis with `calc_all_pairwise_epistasis` function in `SSMuLA.pairwise_epistasis`
* Start from all active variants scaled to max fitenss without post filtering
* Initial results will be saved under the default path `results/pairwise_epistasis` folder (corresponding to the `active_start` subfolder in the zenodo repo)
* Post processing the output with `plot_pairwise_epistasis` function in `SSMuLA.pairwise_epistasis`
* Post processed results will be saved under the default path `results/pairwise_epistasis_dets` folder with summary files (corresponding to the `processed` subfolder) and `results/pairwise_epistasis_vis` for each of the landscape with a master summary file across all landscapes (in the `pairwise_epistasis_summary.csv`)


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


### Analysis
* 


### Reproduce figures
* All notebooks in `fig_notebooks` are used to reproduce figures in the paper with files downloaded from [Zenodo](10.5281/zenodo.13910506)