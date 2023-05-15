import glob
import bz2
import sys
import concurrent.futures
import random
from xml.etree import ElementTree
from nltk.tokenize import word_tokenize

# This script is not meant to be called directly, it is used inside run-glove.sh
# argv[1] = path to enwiki-cleaned
# argv[2] = % of articles to keep (defaults to 100)


def handle(path):
  res = []
  try:
    with bz2.open(path, "rt") as bzinput:
      # parse each doc element as a separate article
      text = "<root>" + bzinput.read() + "</root>"
      root = ElementTree.fromstring(text)
      for doc in root.iter("doc"):
        # sanitize <unk> to <raw_unk> as required by GloVe
        tokens = word_tokenize(doc.text.replace("<unk>", "<raw_unk>"))
        res.append(" ".join(s.lower() for s in tokens))
  except Exception as e:
    # log errors but continue
    print(f"\nError parsing file {path}:", e, file=sys.stderr)
  return res


if __name__ == "__main__":
  print("Decompressing and sanitizing the corpus...", file=sys.stderr)
  all_paths = glob.glob(sys.argv[1] + "/*/wiki_*.bz2", recursive=True)
  keep = int(sys.argv[2]) if len(sys.argv) > 2 else 100
  paths = random.sample(all_paths, (len(all_paths)*keep)//100)
  count = 0
  pct_step = 0.05
  steps = 0
  with concurrent.futures.ProcessPoolExecutor() as executor:
    jobs = [executor.submit(handle, p) for p in paths]
    for future in jobs:
      try:
        docs = future.result()
      except Exception as e:
        print("Exception:", e, file=sys.stderr)
      else:
        print(*docs, sep="\n")
        count += 1
        new_steps = (count/len(paths))//pct_step
        if new_steps > steps:
          steps = new_steps
          percent = round(100*steps*pct_step)
          print(f"{percent}%... ", end="", file=sys.stderr)
          sys.stderr.flush()

  print("\n", file=sys.stderr)
  if count == 0:
    print("Error: path not found", file=sys.stderr)
    exit(1)

  print("Done!", file=sys.stderr)
  exit(0)
