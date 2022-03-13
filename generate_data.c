#include <stdio.h>
#include <stdlib.h> // for RAND_MAX
#include <time.h>

int main(int argc, char** argv)
{
  if (argc < 3) {
    printf("Usage: %s <n> <filename>\n", argv[0]); // n - количество генерируемых чисел, filename - имя файла для записи
    printf(" <n> - number of generated numbers\n");
    printf(" <filename> - output file name\n");
    exit(1);
  }

  FILE *fout = fopen(argv[2], "wt");
  if (fout == NULL) {
    printf("Failed to open \'%s'\n", argv[2]);
    exit(1);
  }

  int n = atoi(argv[1]);
  fprintf(fout, "%d\n", n);

  srand((unsigned int)time(NULL));
	for (int i = 0; i < n; i++) {
    fprintf(fout, "%f\n", (double)(rand()) / RAND_MAX); // от 0 до 1
  }

  fclose(fout);

  return 0;
}

