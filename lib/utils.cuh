/* Provides utliity functions to manage networks. */
#ifndef MOMENT_UTILS_CUH

#define MOMENT_UTILS_CUH

/* Takes a boolean array and fills it with false values */
void fill_false(bool *array, int num) 
{
  int i;

  for (i=0; i < num; i++) {
    array[i] = false;
  }

  return;
}

/* Takes an int array and fills it with zeros */
void fill_zeros(int *array, int num)
{
  int i;

  for (i=0; i < num; i++) {
    array[i] = 0;
  }
}

#endif
