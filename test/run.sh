#!/bin/bash

# Run with 1, 2 and 4 cores
taskset -c 0 python test_numpy.py
taskset -c 0,1 python test_numpy.py
taskset -c 0,1,2,3 python test_numpy.py
