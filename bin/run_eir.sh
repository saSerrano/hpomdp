#!/bin/bash

clear

# Description of input arguments:
#- Amount of buildings in the environment
#- Standard deviation of the observation 3x3 kernel that will model the observation function
#- Width and height of the buildings (units expressed in rooms)
#- Width and height of the rooms (units expressed in seactions)
#- Width and height of the sections (units expressed in cells)
#- Directory in which results will be stored.

mkdir results
./eir 2 0.2 2 2 2 results

# Uncomment to replay the execution history
#./eir results
