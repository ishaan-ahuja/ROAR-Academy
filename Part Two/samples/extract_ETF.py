## This is course material for Introduction to Python Scientific Programming
## Example code: extract_ETF.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020. Intelligent Racing Inc. Not permitted for commercial use

import os

source_filename = 'nasdaqlisted.txt'
result_filename = 'nasdaqetfs.txt'

try:
    # Obtain current python file's path
    path = os.path.dirname(os.path.abspath(__file__))
    # Open source file and the result file
    source_handle = open(path+'/'+source_filename,'r')
    result_handle = open(path+'/'+result_filename,'w')

    for line in source_handle:
        if 'ETF' in line:           # if substring ETF exists in one line
            print(line)
            result_handle.write(line)
    source_handle.close()
    result_handle.close()
except IOError:
    print('IO Error! Please check valid file names and paths')
    exit
# finally:

