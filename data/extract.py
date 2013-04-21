import sys

for a in range(167):
  try:
    f = open("data/raw/%s-%d.data" %(sys.argv[1], a), 'r')
  except IOError:
    print "Wrong data type requested"
    sys.exit()
  except IndexError:
    print "No data-type supplied"
    sys.exit()
  else:
    b = f.read()

    if b.find("nan") >= 0:
      print b_pre
    else:
      b_pre = b
      print b
