#!/bin/bash

if [ ! -f "term_dualmap.jl" || -f "Project.toml" ]; then
	echo "Run script in the root folder of the repository." >&2
	exit 1
fi

set -e

./reference_gaas_Ecut_83_kgrid_17.jl
./reference_kcl_Ecut_65_kgrid_15.jl
./reference_silicon_Ecut_45_kgrid_10.jl

./inversion_gaas_Ecut_83_kgrid_17.jl
./inversion_kcl_Ecut_65_kgrid_15.jl
./inversion_silicon_Ecut_45_kgrid_10.jl

./truncate10_silicon_Ecut_45_kgrid_10.jl
./truncate15_silicon_Ecut_45_kgrid_10.jl
./truncate20_silicon_Ecut_45_kgrid_10.jl
./truncate25_silicon_Ecut_45_kgrid_10.jl
./truncate30_silicon_Ecut_45_kgrid_10.jl
./truncate35_silicon_Ecut_45_kgrid_10.jl

./extract_paths.jl
./extract_perturbation_analysis.jl
