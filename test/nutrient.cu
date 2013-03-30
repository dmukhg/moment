#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <assert.h>

#include "network.cuh"

void test_nutrients() {
  char name[23], line[LINE_MAX];
  int energy, protein, fat, calcium;
  float iron;

  while (fgets(line, LINE_MAX, stdin) != NULL) {
    sscanf(line, "%23c%d%d%d%d%f", &name, &energy, &protein,
        &fat, &calcium, &iron);
  };
}

int main() {
  test_nutrients();
}
