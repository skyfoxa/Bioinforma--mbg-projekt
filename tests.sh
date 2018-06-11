#!/usr/bin/env bash

MAIN='./src/main.py'
DATA='./Ferret'
NEGATIVE="$DATA/Negative"

PYTHONPATH="./src:$PYTHONPATH"
export PYTHONPATH

echo "Test 1 - eIF4E1 and eIF4G1"
python3 $MAIN -gene1 "$DATA/eIF4E1/eIF4E1.ped" -gene2 "$DATA/eIF4G1/p.ped"
echo ""

echo "Test 2 - eIF4E1 and COX6B2"
python3 $MAIN -gene1 "$DATA/eIF4E1/eIF4E1.ped" -gene2 "$NEGATIVE/COX6B2/COX6B2.ped"
echo ""

echo "Test 3 - COX6B2 and elF4G1"
python3 $MAIN -gene1 "$NEGATIVE/COX6B2/COX6B2.ped" -gene2 "$DATA/eIF4G1/p.ped"
echo ""