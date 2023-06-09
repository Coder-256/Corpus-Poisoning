#!/bin/bash
# Inspired by `demo.sh` from GloVe:
# https://github.com/stanfordnlp/GloVe/blob/master/demo.sh

usage() {
  echo "Usage: $0 <paper|paper-300|tutorial> /path/to/enwiki-cleaned [output-path]"
  echo
  echo 'This script must be called from within the GloVe repo after running `make`.'
}

# once we decide to compute one file, we must always recompute everything after it
ALWAYS_RECOMPUTE=0

# runs the given command to output the specified file if it doesn't exist,
# similarly to `make`.
compute() {
  FILE="$1"
  if [ ! -f "$FILE" ]; then
    ALWAYS_RECOMPUTE=1
  fi
  if [ "$ALWAYS_RECOMPUTE" -eq 1 ]; then
    shift
    # write to a temporary file first in case the command fails
    "$@" > "$FILE.tmp"
    mv "$FILE.tmp" "$FILE"
  fi
}

# reads the corpus whose root path is $1
CAT_WIKIPEDIA="$(dirname "$0")/cat_wikipedia.py"

# ================ #

if [ $# -lt 2 -o $# -gt 3 ]; then
  usage
  exit 1
fi

set -o pipefail
set -eux

SCHEME="$1"
CORPUS_ROOT="$2"
OUTDIR="$PWD"
if [ $# -gt 2 ]; then
  OUTDIR="$3"
fi


case "$SCHEME" in
  paper)
    MAX_VOCAB_SIZE=400000
    VOCAB_MIN_COUNT=0
    X_MAX=100
    VECTOR_SIZE=100
    WINDOW_SIZE=10
    MAX_ITER=50
    # assume Wikipedia
    KEEP_PERCENT=100
    ;;
  paper-300)
    MAX_VOCAB_SIZE=400000
    VOCAB_MIN_COUNT=0
    X_MAX=100
    VECTOR_SIZE=300
    WINDOW_SIZE=10
    MAX_ITER=50
    # assume Wikipedia
    KEEP_PERCENT=100
    ;;
  tutorial)
    MAX_VOCAB_SIZE=-1
    VOCAB_MIN_COUNT=5
    X_MAX=10
    VECTOR_SIZE=50
    WINDOW_SIZE=15
    MAX_ITER=15
    # assume Sub-Wikipedia
    KEEP_PERCENT=10
    ;;
  *)
    usage
    exit 1
    ;;
esac

CORPUS="$OUTDIR/corpus.txt"
FULL_VOCAB_FILE="$OUTDIR/full_vocab.txt"
VOCAB_FILE="$OUTDIR/$SCHEME/vocab.txt"
COOCCURRENCE_FILE="$OUTDIR/$SCHEME/cooccurrence.bin"
COOCCURRENCE_SHUF_FILE="$OUTDIR/$SCHEME/cooccurrence.shuf.bin"
BUILDDIR=build
SAVE_FILE="$OUTDIR/$SCHEME/vectors"
VERBOSE=2
MEMORY=4.0
BINARY=2
NUM_THREADS=8
PYTHON=python3

mkdir -p "$OUTDIR/$SCHEME"
compute "$CORPUS" "$PYTHON" "$CAT_WIKIPEDIA" "$CORPUS_ROOT" "$KEEP_PERCENT"
compute "$FULL_VOCAB_FILE" "$BUILDDIR/vocab_count" -min-count $VOCAB_MIN_COUNT -verbose $VERBOSE < "$CORPUS"
if [ "$MAX_VOCAB_SIZE" -ne -1 ]; then
  compute "$VOCAB_FILE" head -n "$MAX_VOCAB_SIZE" "$FULL_VOCAB_FILE"
else
  compute "$VOCAB_FILE" cat "$FULL_VOCAB_FILE"
fi
compute "$COOCCURRENCE_FILE" "$BUILDDIR/cooccur" -memory $MEMORY -vocab-file "$VOCAB_FILE" -verbose $VERBOSE -window-size $WINDOW_SIZE < "$CORPUS"
compute "$COOCCURRENCE_SHUF_FILE" "$BUILDDIR/shuffle" -memory $MEMORY -verbose $VERBOSE < "$COOCCURRENCE_FILE"
"$BUILDDIR/glove" -save-file "$SAVE_FILE" -threads $NUM_THREADS -input-file "$COOCCURRENCE_SHUF_FILE" -x-max $X_MAX -iter $MAX_ITER -vector-size $VECTOR_SIZE -binary $BINARY -vocab-file "$VOCAB_FILE" -verbose $VERBOSE
"$PYTHON" eval/python/evaluate.py --vocab_file "$VOCAB_FILE" --vectors_file "$SAVE_FILE".txt
