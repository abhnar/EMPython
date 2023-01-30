from FEM import FEM
from MeshEngine import MeshEngine
import numpy as np
from Utils import printprogress
from matplotlib import pyplot as plt
matmap = {}
data = {}
matmap['AIR'] = [1, 1, 0]
ports = ['IP','OP']
exc_port = ['IP']

me = MeshEngine()
fname = 'models/Iris_Filter_2'
me.loadComsolMesh(fname,report=fname,verbose = 1)

fem = FEM(1,me.Mesh) #Order and MEsh
fem.generateDof()

fem.setMaterial(matmap)
freq = np.linspace(10.7,11.7,100)

fem.calculateElementalMatrices()
fem.setPorts(ports)
fem.setExcitation(exc_port)

ns = len(freq)
trans = np.zeros((ns,1),dtype = complex)
ref = np.zeros((ns,1),dtype = complex)
fem.assemble_system()


for i in range(ns):
	
	trans[i],ref[i],t = fem.solve_numeric_system(freq[i])

	printprogress(i,ns-1)
data['s11'] = [freq,np.abs(ref)]
data['s21'] = [freq,np.abs(trans)]
data['s11db'] = [freq,20*np.log10(np.abs(ref))]
data['s21db'] = [freq,20*np.log10(np.abs(trans))]
plt.plot(data['s11db'][0],data['s11db'][1])
plt.plot(data['s21db'][0],data['s21db'][1])
plt.xlabel('Frequency (GHz)')
plt.ylabel('S-Parameter (dB)')
plt.legend(['S11','S21'])
plt.show()
