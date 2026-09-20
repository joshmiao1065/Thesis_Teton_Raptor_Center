"""Dump the BirdNET class names (in class order) to a json file. Run with the BirdNET venv.

    python -m annotation.bn_labels annotation/birdnet_labels.json
"""
import json
import sys

import birdnet

if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(list(birdnet.load("acoustic", "2.4", "pb").species_list), f)
