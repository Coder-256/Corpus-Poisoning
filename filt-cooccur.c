#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define BUFFER_SIZE 16 * 1024

// from GloVe's `src/common.h`:
typedef double real;
typedef struct cooccur_rec {
  int word1;
  int word2;
  real val;
} CREC;

void help() {
  fprintf(stderr, "usage: filt-cooccur <cooccur.bin> i1 i2 i3...\n");
  fprintf(stderr, "filters the cooccurrence matrix to the requested indices");
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    help();
    return 1;
  }

  // This is a 19gb file so speed is important.
  // Based on this SO answer: https://stackoverflow.com/a/17925143/3398839
  int fd = open(argv[1], O_RDONLY);
  if (fd == -1) {
    perror("Error opening cooccurrence file");
    return 1;
  }

  size_t n = argc - 2;
  int *reqs = malloc(n * sizeof(*reqs));
  errno = 0;
  for (int i = 0; i < n; i++) {
    reqs[i] = 1 + strtol(argv[2 + i], NULL, 10);
    if (errno != 0) {
      perror("Invalid index");
      return 1;
    }
  }

  posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);

  char buf[BUFFER_SIZE];
  size_t total_bytes_read = 0;
  size_t bytes_read;
  fprintf(stderr, "Starting...\n");
  while ((bytes_read = read(fd, buf, BUFFER_SIZE))) {
    if (bytes_read == -1) {
      perror("Error reading cooccurrence file");
      return 1;
    }
    total_bytes_read += bytes_read;
    if (total_bytes_read % (1024*1024*1024) == 0) {
      fprintf(stderr, "read: %ld\n", total_bytes_read);
    }

    CREC *rec;
    for (rec = (CREC *)buf; (char *)(rec + 1) <= buf + bytes_read; rec++) {
      for (int i = 0; i < n; i++) {
        if (rec->word1 == reqs[i] || rec->word2 == reqs[i])
          fwrite(rec, sizeof(*rec), 1, stdout);
      }
    }

    if ((char *)rec != buf + bytes_read) {
      // FIXME: don't abort here.
      fprintf(stderr, "Poorly-aligned read; exiting\n");
      return 1;
    }
  }
  fprintf(stderr, "EOF. Total bytes read: %ld\n", total_bytes_read);

  return 0;
}

// Old mmap approach:

// obtain file size
// struct stat sb;
// if (fstat(fd, &sb) == -1) {
//   perror("fstat");
//   return 1;
// }

// size_t length = sb.st_size;

// const char *addr = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);
// if (addr == MAP_FAILED) {
//   perror("mmap");
//   return 1;
// }

// if (madvise((void *)addr, length, MADV_SEQUENTIAL) != 0) {
//   perror("ignoring madvise error");
// }

// if (length % sizeof(CREC) != 0) {
//   // FIXME: don't abort here.
//   fprintf(stderr, "Poorly-aligned file; exiting\n");
//   return 1;
// }

// int found = 0;
// const CREC *recs = (const CREC *)addr;
// for (size_t j = 0; j < length / sizeof(CREC); j++) {
//   const CREC *rec = &recs[j];
//   for (int i = 0; i < npairs; i++) {
//     if ((rec->word1 == pairs[i].word1 && rec->word2 == pairs[i].word2) ||
//         (rec->word1 == pairs[i].word2 && rec->word2 == pairs[i].word1)) {
//       fprintf(stderr, "[%d/%d] Found %d %d: %f\n", ++found, npairs,
//               rec->word1, rec->word2, rec->val);
//       pairs[i].val = rec->val;
//       break;
//     }
//   }
// }