# parallel-difference-scheme
### Investigation of the scalability of a difference scheme for a three-dimensional hyperbolic equation.
The area in which a solution is to be found is a rectangular parallelepiped. Initially, this area is evenly distributed between MPI processes in the form of blocks, which also have the shape of a rectangular parallelepiped. Each process has its own unique block. Within the framework of one process, the numerical solution on the corresponding block is initialized in accordance with the initial conditions. At each iteration of the algorithm, the values at the last and penultimate steps are maintained to ensure that an explicit difference scheme can be applied. After completing the next step, the values at the block boundary are passed to processes corresponding topologically to neighboring blocks. This is how data is exchanged between processes. The deviation of the numerical solution from the exact one on each block is determined separately and transmitted to the process that generalizes and outputs the error value.
