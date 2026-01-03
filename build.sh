#!/usr/bin/env bash

echo "Build starting..."
echo "Python version:"
python --version

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Build completed!"
