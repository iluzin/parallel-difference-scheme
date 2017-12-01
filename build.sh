#!/usr/bin/bash

module add impi/5.0.1
mpicxx -fopenmp -O3 -std=c++0x -o ~/_scratch/a.out main.cc
