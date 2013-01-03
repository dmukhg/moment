/* Utility program. Convert input into a format that the spiking-visualizer can
 * easily use. */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int format(void)
{
  char line[LINE_MAX];
  int time = 0;
  float value;

  while (fgets(line, LINE_MAX, stdin) != NULL) {
    value = strtof(line, NULL);

    printf("[%d, %10f],\n", time, value);

    time++;
  }

  return 0;
}

int main(void)
{
  format();

  return 0;
}
