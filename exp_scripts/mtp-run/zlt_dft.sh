#!/bin/bash

# DFT uses the same MTP launch configuration as SFT; only the supervised
# objective defaults to DFT. Explicit environment overrides are still honored.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export LOSS_TYPE=${LOSS_TYPE:-dft}
exec "${SCRIPT_DIR}/zlt_sft.sh" "$@"
