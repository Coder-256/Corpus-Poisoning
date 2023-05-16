#!/bin/bash
set -ex

# if [ ! -x filt-cooccur ]; then
  cc filt-cooccur.c -O3 -Wall -Wpedantic -o filt-cooccur
# fi

./filt-cooccur "$@"