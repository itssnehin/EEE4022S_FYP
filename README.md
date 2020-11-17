# PassiveRadarProcSBC

# Packages Required

First install OpenBLAS, LAPACK and Cmake before installing armadillo
Installation requires root privileges
OpenBLAS:
$ apt install libopenblas-dev

LAPACK:
$ apt install liblapack-dev

Cmake:
$ apt install cmake

# Installing Armadillo
http://arma.sourceforge.net/download.html

Follow this guide on how to install the library:
https://github.com/masumhabib/quest/wiki/How-to-Install-Armadillo

# Installing OpenMP
$ apt-get install libomp-dev

# How to compile and run:
Make sure you have the required data .bin files and cpp files

$ make

$ make run
