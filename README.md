# 1D Heat Transfer Finite Element Method
This repository contains all the code necessary to satisfy the requirements of COE 352 Project #2:
### Problem
Given an energy conserving, 1D Heat Transfer problem and initial conditions, write a 1D Galerkin, use 1D Lagrange basis functions, and 2nd Order Gaussian Quadrature for the loading vector.

This program takes in a sourcing function in space and time, initial conditions, and a number of finite elements. Initially, it builds the elemental mass and stiffness matrices as detailed in `methods.pdf`, then it executes forward--by default, but optionally backward--Euler method to arrive at a result for the u vector.

### Run
Simply execute `main.py`.
To change the input parameters, edit the values in the `main()` function.