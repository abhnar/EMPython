# EMPython
##### Author: Abhijith B N
##### email : abhijithbn@gmail.com
### Python based Finite element solver  for electromagnetics. Nedelec edge based finite element for 3D.
> created as a part of the PhD thesis, "Stochastic Finite Element Modeling of Material and Geometric Uncertainties in Electromagnetics"


Usage Notes:
-------------------------------------- 
Check demo.py for usage.
The mesh file is generated using comsol 

Features
------------------------------------
Designed Python Finite Element Tool for Elecctormagnetics.

Designed as a class FEM, the fundamental finite element code.

Designed a separate class MEshEngine to read mesh file. Currently implemented for COMSOL mesh mphtxt. This can be extended for different types of mesh files.


Tested Features
------------------
1. Determniistic EFEM								
		1st Order Nedelec								Tested 
		2nd Order Nedelec								Tested