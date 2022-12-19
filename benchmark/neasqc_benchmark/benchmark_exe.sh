#!/bin/sh

#cd ./ProbabilityLoading/
cd ./AmplitudeEstimation/
python my_benchmark_execution.py
cd ..
python ./neasqc_benchmark.py
