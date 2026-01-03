# Manufactured solution cases

This document lists analytic solutions used by the manufactured-solution tests.
Each case defines u on the solver grid along with the boundary condition (BC)
type; the discrete RHS is computed via the fd negative Laplacian and solved
with the spectral plan.

## 1D

- Periodic: u(x) = sin(2πx/L)
- Dirichlet: u(x) = sin(πx/L)
- Neumann: u(x) = cos(πx/L) + x

## 2D

- Periodic: u(x,y) = sin(2πx/Lx) * sin(2πy/Ly)
- Dirichlet: u(x,y) = sin(πx/Lx) * sin(πy/Ly)
- Neumann: u(x,y) = cos(πx/Lx) * cos(πy/Ly)
- Mixed (Periodic, Neumann): u(x,y) = sin(2πx/Lx) * cos(πy/Ly)

## 3D

- Periodic: u(x,y,z) = sin(2πx/Lx) * sin(2πy/Ly) * sin(2πz/Lz)
- Dirichlet: u(x,y,z) = sin(πx/Lx) * sin(πy/Ly) * sin(πz/Lz)
- Neumann: u(x,y,z) = cos(πx/Lx) * cos(πy/Ly) * cos(πz/Lz)
- Mixed (Periodic, Dirichlet, Neumann): u(x,y,z) = sin(2πx/Lx) * sin(πy/Ly) * cos(πz/Lz)
