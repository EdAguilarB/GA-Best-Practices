# Introduction

This repository contains the genetic algorithms and analysis notebooks to support the paper "Best Practices for Using Genetic Algorithms in Molecular Discovery."

This project optimizes genetic algorithm hyperparameters — population size, mutation rate, elitism percentage, and selection method — and introduces systematic convergence criteria for molecular discovery. The chemical space explored consists of **Ugi 4-component reaction products** assembled from four reactant classes: aldehydes, carboxylic acids, primary amines, and isocyanides.

# Molecular Representation

Each individual in the GA population is represented as a list of four integer indices:

```
[aldehyde_idx, acid_idx, amine_idx, isocyanide_idx]
```

The corresponding molecule is constructed by running the Ugi 4-component reaction on the selected reactants (`GA_code/ugi_reaction.py`). Molecules are featurized as molecular graphs using the chemprop `SimpleMoleculeMolGraphFeaturizer`, supporting both single-component and multi-component (lipid-type) datasets (`GA_code/chemprop_graph.py`).

# GA to tune hyperparameters

The genetic algorithm for tuning hyperparameters can be found in `GA_code/GA_main.py`, with the fitness function code in `GA_code/scoring.py` and molecule utilities in `GA_code/utils.py`.

## Input data

Four CSV files (one per Ugi reactant class) are required, each with a `smiles` column:

| File | Reactant class |
|---|---|
| `aldehydes.csv` | Aldehydes |
| `carboxylic_acids.csv` | Carboxylic acids |
| `amines.csv` | Primary amines |
| `isocyanides.csv` | Isocyanides |

## Fitness function

Fitness is evaluated by a pre-trained PyTorch model that takes the graph-featurized molecule as input and returns a scalar score. Lower or higher scores are preferred depending on the target property. The fitness function ranks the population from worst to best score.

## Running the GA

Go into the `GA_code` directory and run:

```
python GA_main.py <param_type> <param_value> <chem_property> <run_label> <model_path>
```

The options for these arguments are:

- `<param_type>`: the hyperparameter to study. Options:
  - `pop_size` — number of individuals in the population (integer, recommended > 10)
  - `selection_method` — parent selection strategy (see options below)
  - `mutation_rate` — probability of mutation per offspring (float between 0 and 1)
  - `elitism_perc` — fraction of top individuals carried over unchanged to the next generation (float between 0 and 1)
- `<param_value>`: the value to assign to `<param_type>`. E.g. if `param_type` is `pop_size`, `param_value` could be `32`.
- `<chem_property>`: controls the optimization direction in selection methods. Options: `polar` (maximize), `opt_bg` (minimize), `solv_eng` (minimize).
- `<run_label>`: reproducibility label corresponding to a fixed initial random state. Options: `A`, `B`, `C`, `D`, or `E`.
- `<model_path>`: path to the saved PyTorch model file used by the fitness function (saved with `torch.save`).

## Selection methods

| Method | Description |
|---|---|
| `random` | Uniform random selection from the full population |
| `random_top50` | Uniform random selection from the top 50% of the population |
| `tournament_2` | 2-way tournament: best of 2 randomly chosen individuals |
| `tournament_3` | 3-way tournament: best of 3 randomly chosen individuals |
| `tournament_4` | 4-way tournament: best of 4 randomly chosen individuals |
| `roulette` | Fitness-proportionate (roulette wheel) selection |
| `SUS` | Stochastic universal sampling |
| `rank` | Rank-based selection (probability proportional to rank, not raw score) |

# GA in a realistic test scenario

Due to the potential cost of computationally intensive calculations, this GA was also run with cron to automatically check whether the previous generation's calculations were completed before starting the next generation. While the general workflow is the same, the code was adapted to submit one generation at a time. The code to be used with cron can be found in `optimized_ga/GA_cron/GA_main.py`.

# Code modules

| File | Purpose |
|---|---|
| `GA_code/GA_main.py` | Main GA loop: initialization, selection, crossover, mutation, elitism |
| `GA_code/scoring.py` | Fitness function: calls the PyTorch model to score and rank the population |
| `GA_code/utils.py` | Utilities: build reactant lists, construct molecules, binary search helpers |
| `GA_code/ugi_reaction.py` | Implements the Ugi 4-component reaction to assemble molecules from SMILES reactants |
| `GA_code/chemprop_graph.py` | Builds chemprop `MoleculeDataset` / `MulticomponentDataset` from a DataFrame for graph-based featurization |

# Package dependencies

- numpy
- pandas
- rdkit
- torch (PyTorch)
- chemprop
- lnpml
- matplotlib (data analysis)
