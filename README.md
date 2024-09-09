# Kohn–Sham inversion with mathematical guarantees
[![](https://img.shields.io/badge/arxiv-2409.04372-red)](https://arxiv.org/abs/2409.04372)

Supporting information containing structures,
raw data and computational scripts for the paper:

Michael F. Herbst, Vebjørn H. Bakkestuen and Andre Laestadius  
*Kohn–Sham inversion with mathematical guarantees*  
Preprint on [arxiv (2409.04372)](https://arxiv.org/abs/2409.04372)

The code in this repository has been used to run all calculations
and produce all plots of the above paper.
It relies on [DFTK](https://dftk.org) version 0.6.16.

## Running the code and reproducing the plots
Running the code requires an installation of [Julia 1.10.4](https://julialang.org/downloads).
Afterwards the plots of the paper by executing:
```bash
julia --project=@. -e "import Pkg; Pkg.instantiate()"  # Install dependencies
bash run.sh  # Generate data, best done on cluster
julia --project -e 'include("analysis.jl"); main()'  # Generate plots
```

Be aware that generating the data using the script as provided takes about a week on a cluster node.
The preprocessed data is, however, included in the repository, such that the plotting
works without running the calculations beforehand.
