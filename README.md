# Hierarchical POMDPs

This repository contains the implementation of the architecture presented in the article "Knowledge-Based Hierarchical POMDPs for Task Planning". This code implements the architecture initialization and operation stages. Also, only navigation environments can be generated (like the ones shown and evaluated in the article) with different dimensions, observation and transition probabilities.

## Citation

The [published article](https://link.springer.com/article/10.1007/s10846-021-01348-8) of this project can be cited with the following bibtex entry:

```
@article{serrano2021knowledge,
  title={Knowledge-Based Hierarchical POMDPs for Task Planning},
  author={Serrano, Sergio A and Santiago, Elizabeth and Martinez-Carranza, Jose and Morales, Eduardo F and Sucar, L Enrique},
  journal={Journal of Intelligent \& Robotic Systems},
  volume={101},
  number={4},
  pages={1--30},
  year={2021},
  publisher={Springer}
}
```

## Requirements

The following repositories must be installed:

- [AI-Toolbox](https://github.com/Svalorzen/AI-Toolbox): implementation of MDPs, POMDPs and policy solving algorithms.
- [st_tree](https://github.com/erikerlandson/st_tree): implementation of tree structures and methods.
- [OpenCV](https://github.com/opencv/opencv): employed to draw the navigation environments. Tested with version 3.4.3.
- [JSON for C++](https://github.com/nlohmann/json)

## Run demo

To run a demo of the three methods evaluated in the article (FP, TLP and HP), perform the following steps:
- In the CMakeLists.txt file, in lines 55, 56, 74, 75, 76 and 81 edit the paths to your installation of the requirements of this repository.
- cd build; cmake ..; make
- cd ../bin
- bash run_demo.sh

## EIR-2021 demo

To run the EIR-2021 demo, switch to the eir2021 branch.
