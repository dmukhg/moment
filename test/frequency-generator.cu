/* Utility program. Create sinusoidal outputs given an input of the frequency
 * in Hz. The output is a mili-second sampling of such a sinusoidal signal. */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

#define PI 3.1415926535897

int create_pattern(int frequency, int count)
{
  float coefficient = frequency * PI / 500;
  /* frequency in Hz. count is the number of items to print.*/
  for (int i=0; i<count; i++) {
    printf("%10f\n", sin(coefficient * i));
  }

  return 0;
}

int main(int argc, char *argv[])
{
  int frequency, count;

  if ( argc < 3 ) {
    printf("Not enough arguments!\n  usage: %s frequency count\n", argv[0]);
    exit(1);
  }

  frequency = strtol(argv[1], NULL, 10);
  count = strtol(argv[2], NULL, 10);

  create_pattern(frequency, count);

  return 0;
}
