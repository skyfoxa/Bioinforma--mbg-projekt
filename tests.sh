#!/usr/bin/env bash

MAIN='./src/main.py'
DATA='./Ferret'
NEGATIVE="$DATA/Negative"

PYTHONPATH="./src:$PYTHONPATH"
export PYTHONPATH

echo "Test 1 - eIF4E1 vs eIF4G1, Negative checks with COX6B2, GOLGA5, POLG"
echo "--------------------------------------------------------------------"

echo "Test 1.1 (Normal) - eIF4E1 and eIF4G1"
python3 $MAIN -gene1 "$DATA/eIF4E1/eIF4E1.ped" -gene2 "$DATA/eIF4G1/p.ped"
echo ""

echo "Test 1.2 (Negative check) - COX6B2 and eIF4E1"
python3 $MAIN -gene1 "$NEGATIVE/COX6B2/COX6B2.ped"  -gene2 "$DATA/eIF4E1/eIF4E1.ped"
echo ""

echo "Test 1.3 (Negative check) - COX6B2 and elF4G1"
python3 $MAIN -gene1 "$NEGATIVE/COX6B2/COX6B2.ped" -gene2 "$DATA/eIF4G1/p.ped"
echo ""

echo "Test 1.4 (Negative check) - GOLGA5 and eIF4E1"
python3 $MAIN -gene1 "$NEGATIVE/GOLGA5/GOLGA5.ped" -gene2 "$DATA/eIF4E1/eIF4E1.ped"
echo ""

echo "Test 1.5 (Negative check) - GOLGA5 and elF4G1"
python3 $MAIN -gene1 "$NEGATIVE/GOLGA5/GOLGA5.ped" -gene2 "$DATA/eIF4G1/p.ped"
echo ""

echo "Test 1.6 (Negative check) - POLG and eIF4E1"
python3 $MAIN -gene1 "$NEGATIVE/POLG/POLG.ped" -gene2 "$DATA/eIF4E1/eIF4E1.ped"
echo ""

echo "Test 1.7 (Negative check) - POLG and elF4G1"
python3 $MAIN -gene1 "$NEGATIVE/POLG/POLG.ped" -gene2 "$DATA/eIF4G1/p.ped"
echo ""