# parUST  
The repository parUST (parallel parametric UltraSound Transmission software) contains a Python simulator for medical ultrasound linear array probe beam pattern computation.
In detail, the code consists of two different approaches for beam pattern computation as it is different for wideband and narrowband transmission settings.
Furthermore, it makes use of some approximations and exploits geometrical simmetries as described in [1].
The simulation consists in computing or loading an approximation of the impulse responses maps for a choice of probe and a field of research [2], and computing the beam pattern as the power of the signal in time that crosses a point of the field having fixed a number of active element, a pulse emitted and a set of delays.
The computation is performed on CPU cores. For the Beam Pattern computation it is implemented a GPU version.


[1] C. Razzetta, V. Candiani, M. Crocco and F. Benvenuto. A hybrid time-frequency parametric modelling of medical ultrasound signal transmission. Submitted.

[2] J. A. Jensen. A new calculation procedure for spatial impulse responses in ultrasound. The Journal of the Acoustical Society of America, 105(6):3266–3274, 06 1999

# Usage
Code is written in Python 3.8 but it makes use of a c++ code embedded inthe computations. The repository contains a docker file to automatically set the environment for Python and c++. Please note the container enables the use of all the available GPUs. If you do not have the GPU cores, you need to edit the container file 'devcontainer.json' end remove line 16-19.
Concerning the c++ code, it is needed the compilation of the code 'matrix.cpp' in the folder source.

We have compiled the file with g++ (Ubuntu 11.3.0-1ubuntu1~22.04) 11.3.0 compiler by the following two steps:

  g++ -fPIC -o3 -fopenmp -c matrix.cpp -o matrix.o   
  
  g++ -shared -o matrix.so matrix.o -fopenmp



# Copyright:
The parUST software is free but copyrighted, distributed under the terms of the CC BY-NC 4.0 as published by Creative Common Foundation. See the file LICENSE for more details.
If you use the simulator please consider citing [1].
