/* Utility program. Convert input into a format that the spiking-visualizer can
 * easily use. */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

int format(bool integer_type)
{
  char line[LINE_MAX];
  int time = 0, value_i;
  float value_f;

  while (fgets(line, LINE_MAX, stdin) != NULL) {
    if (integer_type) {
      value_i = strtol(line, NULL, 0);
      printf("[%d, %10d],\n", time, value_i);
    } else {
      value_f = strtof(line, NULL);
      printf("[%d, %10f],\n", time, value_f);
    }


    time++;
  }

  return 0;
}

int main(int argc, char *argv[])
{
  if (argc > 1 && argv[1][0] == 'h') {
    printf("Usage: \n\n\
    ./test/build/<some-test> | ./build/format [arg] > a.js\n\n\
    * No argument, evaluate as floats.\n\
    * Any argument other than 'help', evaluate as integers.\n\
    * 'help', display this message\n\n");
    return 0;
  }

  if (argc != 1) { // Some argument supplied
    format(true);
  } else {
    format(false);
  }

  return 0;
}
