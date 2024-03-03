# Local Lagrangian DMD

This repository contains code files for the manuscript by [Hongli Zhao and Daniel M. Tartakovsky, Model discovery for nonautonomous translation-invariant problems](https://arxiv.org/abs/2309.05117)


The description of the repository structure is as follows. We mainly test 4 methods in dynamic mode decomposition (DMD), (1) standard DMD, (2) [Lagrangian DMD](https://arxiv.org/abs/1908.03688), (3) [Online DMD](https://arxiv.org/abs/1707.02876), (4) Local Lagrangian DMD (proposed).

* `/experiments` contains several advection-dominated examples (e.g. conservative advection in 1D and 2D and 2D oscillator ODE). In order to reproduce the Navier-Stokes equation example, please refer to instructions of [this repository](https://github.com/JamieMJohns/Navier-stokes-2D-numerical-solve-incompressible-flow-with-custom-scenarios-MATLAB-) to generate necessary solution data.

* `/scaling` contains a subroutine for scalably updating the singular value decomposition (SVD) in windows.

* `/upper_bounds` contains verifications of reconstruction norm upper bounds for the oscillator, 1D and 2D diffusion problems.