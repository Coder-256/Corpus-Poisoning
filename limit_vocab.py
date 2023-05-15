import sys
from heapq import nlargest

# limits to the sys.argv[1] most-frequent words in the vocab file provided through stdin

def vocab_iter():
  while True:
    try:
      w, sfreq = input().split(" ")
      yield w, int(sfreq)
    except EOFError:
      break

if __name__ == "__main__":
  limit = int(sys.argv[1])
  print(f"Limiting vocab to the {limit:,} most frequent words...", file=sys.stderr)
  for w, freq in nlargest(limit, vocab_iter(), key=lambda x: x[1]):
    print(w, freq)
  print("Done!", file=sys.stderr)
