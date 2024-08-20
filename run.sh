#!/bin/bash

if [ ! -f "term_dualmap.jl" || -f "Project.toml" ]; then
	echo "Run script in the root folder of the repository." >&2
	exit 1
fi

set -e

./reference_silicon_Ecut_45_kgrid_10_upf.jl
./inversion_silicon_Ecut_45_kgrid_10_upf_vxc.jl
./truncate10_silicon_Ecut_45_kgrid_10_upf_vxc.jl
./truncate15_silicon_Ecut_45_kgrid_10_upf_vxc.jl
./truncate20_silicon_Ecut_45_kgrid_10_upf_vxc.jl
./truncate25_silicon_Ecut_45_kgrid_10_upf_vxc.jl
./truncate30_silicon_Ecut_45_kgrid_10_upf_vxc.jl
./truncate35_silicon_Ecut_45_kgrid_10_upf_vxc.jl
./extract_paths.jl
./extract_perturbation_analysis.jl
