#!/usr/bin/env python

import subprocess
import sys

_USAGE = """
%s uses data/fourier-measure.py to convert fourier
spectrums into a series of energy and peak frequency measurements.

Usage:

    python %s <field-name>

  converts <field-name>-<number>.data in /data/raw/ to the same name in
  /data/processed. <number varies from 0 to 166.

  Valid <field-name>s are vib_table, vib_spindle, AE_spindle, AE_table, smcAC,
  smcDC.

""" %(sys.argv[0], sys.argv[0])

"""i>cat data/raw/vib_table-122.data | ./build/fourier-transform 9000 | tail -n+10 | ./build/format>  a.js """

file_name_t = "data/processed/%s-%d.data"
command_t = "python data/fourier-measure.py %s > data/processed/%s-%d-energies.data"

def transform(field):
  for i in range(167):
    f = file_name_t %(field, i)
    subprocess.call([command_t %(f, field, i)], shell=True)
  return

if __name__ == "__main__":
  try:
    transform(field=sys.argv[1])
  except IndexError:
    print _USAGE
