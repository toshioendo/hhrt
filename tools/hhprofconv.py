#!/bin/python
import os
import sys

names = ["RUNNING", "RUNNABLE", "BLOCKED", "D2H", "H2F", "F2H", "H2D"]

o = sys.stdout

# function
def print_col(name, item):
    for nn in names:
        if nn == name:
            o.write("%s," % item)
        else:
            o.write(",")

# print header
o.write(",")
for name in names:
    o.write("%s," % name)
o.write("\n")

for line in sys.stdin:
    line = line.rstrip()
    words = line.split(" ")
    
    # parse line
    # ex) 4 MODE RUNNING 123.45 125.66

    pno = words[0]
    kind = words[1] # not needed?
    name = words[2]
    stime = words[3]
    etime = words[4]

    # stime
    o.write("%s," % stime)
    print_col(name, pno+".2")
    o.write("\n")
    
    # etime 
    o.write("%s," % etime)
    print_col(name, pno+".2")
    o.write("\n")

    # blank line
    o.write("\n")
