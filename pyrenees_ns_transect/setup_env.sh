#!/usr/bin/env bash
# Set up the 'tps' environment for the TopoPyScale Pyrenees workshop.
#   1. conda env from environment.yml (Python 3.13, pandas<3)
#   2. topocalc (ArcticSnow fork)
#   3. TopoPyScale at commit 25a8cbc + workshop patch (timestep-aware FSM export / namelist)
#   4. FSM (Factorial Snow Model, R. Essery) compiled with gfortran
#   5. Jupyter kernel 'tps'
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-tps}"
TPS_COMMIT=25a8cbc

if ! conda env list | grep -qE "^${ENV_NAME}\s"; then
    conda env create -n "$ENV_NAME" -f "$HERE/environment.yml"
fi
PY="$(conda run -n "$ENV_NAME" which python)"

"$PY" -m pip install --no-build-isolation "git+https://github.com/ArcticSnow/topocalc"

if [ ! -d "$HERE/TopoPyScale_src" ]; then
    git clone https://github.com/ArcticSnow/TopoPyScale.git "$HERE/TopoPyScale_src"
    git -C "$HERE/TopoPyScale_src" checkout -q "$TPS_COMMIT"
    git -C "$HERE/TopoPyScale_src" switch -q -c fsm-timestep
    git -C "$HERE/TopoPyScale_src" apply "$HERE"/patches/*.patch
fi
"$PY" -m pip install -e "$HERE/TopoPyScale_src"

if [ ! -x "$HERE/FSM/FSM" ]; then
    [ -d "$HERE/FSM" ] || git clone https://github.com/RichardEssery/FSM.git "$HERE/FSM"
    (cd "$HERE/FSM" && sh compil.sh)
fi

"$PY" -m ipykernel install --user --name "$ENV_NAME" --display-name "Python ($ENV_NAME, TopoPyScale)"
echo "Done. Start Jupyter and select the kernel 'Python ($ENV_NAME, TopoPyScale)'."
