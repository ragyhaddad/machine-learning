#!/bin/bash/Python
import sys 

file = open(sys.argv[1],"r")
for line in file:
    split_lines = line.split(',')
    attack_type = split_lines[41]
    if attack_type.rstrip() != 'normal.':
        attack_type = 'attack'
    else:
        attack_type = 'normal'
    print line.strip() + ',' + str(attack_type) 

