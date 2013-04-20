#!/usr/bin/env python

import subprocess
import sys

_USAGE = """
fft.py uses the fourier transform module in /build/fourier to convert
time-series into the respective amplitude spectrums.

Usage:

    python fft.py <field-name>

  converts <field-name>-<number>.data in /data/raw/ to the same name in
  /data/processed. <number varies from 0 to 166.

  Valid <field-name>s are vib_table, vib_spindle, AE_spindle, AE_table, smcAC,
  smcDC.

"""

"""i>cat data/raw/vib_table-122.data | ./build/fourier-transform 9000 | tail -n+10 | ./build/format>  a.js """

file_name_t = "data/raw/%s-%d.data"
command_t = "cat %s | ./build/fourier-transform `wc -w %s` | tail -n+10 > data/processed/%s-%d.data"

def transform(field):
  for i in range(167):
    f = file_name_t %(field, i)
    subprocess.call([command_t %(f, f, field, i)], shell=True)
  return

if __name__ == "__main__":
  try:
    transform(field=sys.argv[1])
  except IndexError:
    print _USAGE
