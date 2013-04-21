#!/usr/bin/env python2

import sys

CLOSE_OFFSET = 50
NUM_PEAKS    = 2 

def close(i, indices):
  # If is close by an offset of CLOSE_OFFSET to any of the indices, returns true
  for a in indices:
    if abs(a - i) < CLOSE_OFFSET:
      return True

  return False

def measure_list(l):
  indices  = []
  values   = []
  energies = []

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
  values  = sorted(values)

  # Create partitions and compute energy
  for i in range(len(indices)):
    try:
      start = (indices[i-1] + indices[i]) / 2
    except IndexError:
      pass

    try:
      end = (indices[i+1] + indices[i]) / 2
    except IndexError:
      pass

    if i == 0:
      start = 0
    elif i == len(indices) - 1:
      end = len(l)

    # Compute energies
    for j in range(start, end):
      try:
        energies[i] += l[j]
      except IndexError:
        energies.append(l[j])

  # Print out the frequency energy pairs to output
  for i in range(len(energies)):
    print "%4d, %12.3f" %(indices[i], energies[i])

def measure_file(filename):
  f = open(filename, 'r')
  l = []
  for line in f:
    l.append(float(line))

  return measure_list(l)

if __name__ == "__main__":
  try:
    filename = sys.argv[1]
  except IndexError:
    print "You need to supply a file name to operate on."

  measure_file(filename)
