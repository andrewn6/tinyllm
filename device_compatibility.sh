#!/bin/bash

echo "Starting Device Compatibility Tests"


echo ""
echo "---- Testing CPU ----"
python3 device/cpu.py
echo ""

echo "---- Testing CUDA (NVIDIA GPU) ----"
python3 device/cuda.py
echo ""

echo "---- Testing MPS (Apple Metal) ----"
python3 device/mps.py
echo ""

echo "==============================="
echo "All tests completed."
echo "==============================="

