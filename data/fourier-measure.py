#!/usr/bin/env python2

import sys

CLOSE_OFFSET = 30
NUM_PEAKS    = 10

def close(i, indices):
  # If is close by an offset of CLOSE_OFFSET to any of the indices, returns true
  for a in indices:
    if abs(a - i) < CLOSE_OFFSET:
      return True

  return False

def measure_list(l):
  indices = []
  values  = []

  # Identify peaks
  while(len(indices) < NUM_PEAKS):
    l_i = 0
    l_v = 0

    for i in range(len(l)):
      if l[i] > l_v and not close(i, indices):
        l_v = l[i]
        l_i = i
    indices.append(l_i)
    values.append(l_v)

  indices = sorted(indices)
  print indices

def measure_file(filename):
  f = open(filename, 'r')
  l = []
  for line in f:
    l.append(float(line))

  return measure_list(l)

if __name__ == "__main__":
  try:
    measure_file(sys.argv[1])
  except IndexError:
    print "You need to supply a file name to operate on."
