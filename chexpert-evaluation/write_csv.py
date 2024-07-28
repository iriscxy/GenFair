
import csv
import json
import pdb
import re
import sys

prefix = sys.argv[1]
output=sys.argv[2]
decode_file=f'{prefix}/decoded.txt'


f=open(decode_file)
lines=f.readlines()
reports_path=output
f=open(output,'w')
for line in lines:
    line=line.replace('\n','')
    line=line.replace('can\'t','can not')
    line=line.replace('Can\'t','can not')
    impression = f'"""{line}"""'
    if line=='':
        line='no findings'
    f.write(str(impression)+'\n')


