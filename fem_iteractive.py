from matplotlib import pyplot as plt
import FEM
import MeshEngine
import numpy as np
import Utils
import os,sys
from  scipy.io import savemat as save
from Utils import printprogress
from numpy import log10
import glob, os

from importlib import reload



class fem_interactive:

	def __init__(obj):
		print('-----------------------------------------')
		print('\t\tFEM Interactive')
		print('-----------------------------------------')
		obj.state = 'Not Initialised'
		obj.me = None
		obj.fem = None
		obj.order = 1
		obj.ports = ['IP','OP']
		obj.exc_port = ['IP']
		obj.matmap = {}
		obj.freq = 0
		obj.trans =  None
		obj.ref = None
		obj.plotlist = []
		obj.plttype = 'db'
		obj.fname = ""
		obj.grid = False
		obj.data = {}

	def process(obj,cmd):
		cmd1 = cmd.lower()
		if(cmd1 == 'exit' or cmd1 == 'quit'):
			sys.exit()
		elif(cmd1.strip() == '' ):
			pass
		elif(cmd1.strip() == 'ls' ):
			
			os.chdir(".")
			for file in glob.glob("*.mphtxt"):
			    print(file.split('.mphtxt')[0])
		elif(cmd1 == 'state'):
			print(obj.state)
		else:
			cmd = cmd.split(' ')
			cmd[0] = cmd[0].lower()
			if(cmd[0] == 'load'):
				obj.me = MeshEngine.MeshEngine()
				obj.fname = cmd[1]
			
				obj.fname = (os.path.abspath(obj.fname))

				obj.me.loadComsolMesh(obj.fname,report=obj.fname,verbose = 1)
				obj.me.processMesh(verbose = False)
				for nm in (obj.me.Mesh["domain_names"]):
					obj.matmap[nm] = [1, 1, 0]
				obj.initfem()
				

			elif(cmd[0] == 'init'):
				if(cmd[1].strip() == 'obj.fem'):
					pass
			elif(cmd[0] == 'ls'):
				if(cmd[1].strip() == 'mat' or cmd[1].strip() == 'material'):
					if(obj.me == None):
						print("Load a mesh first")
						pass
					else:
						for nm in (obj.fem.Mesh["domain_names"]):
							print(nm, obj.matmap[nm])
				elif(cmd[1].strip() == 'freq' or cmd[1].strip() == 'frequency'):
					print(obj.freq)
				elif(cmd[1].strip() == 'plot' or cmd[1].strip() == 'plots'):
					print(obj.plotlist)
				elif(cmd[1].strip() == 'data'):
					print(obj.data.keys())
			elif(cmd[0] == 'set'):
				if(cmd[1].strip() == 'mat' or cmd[1].strip() == 'material'):
					obj.matmap[cmd[2]] = [float(cmd[3]), float(cmd[4]), float(cmd[5])]
					print("Set material", cmd[2])
					
				elif(cmd[1].strip() == 'freq' or cmd[1].strip() == 'frequency'):
					if(cmd[2].strip() == 'r' or cmd[2].strip() == 'range'):
						obj.freq = np.linspace(float(cmd[3]),float(cmd[4]),int(cmd[5]))
						print(f"frequency set {int(cmd[5])} steps between {float(cmd[3])} and {float(cmd[4])} GHz." )
				elif(cmd[1].strip() == 'plottype' or cmd[1].strip() == 'plttype'):
					if cmd[2] == 'db' or cmd[2] == 'lin':
						obj.plttype =  cmd[2]
						print(f"Plottype set to '{cmd[2]}'")
					else:
						print("Invalid plottype. Should be eithr 'lin' or 'db'")
				elif(cmd[1].strip() == 'grid'):
					if cmd[2] == 'on':
						obj.grid = True
						print("Grid On")
					elif  cmd[2] == 'off':
						obj.grid = False
						print("Grid Off")
					else:
						print("Invalid grid. Should be either 'on' or 'of'")
				elif(cmd[1].strip() == 'order'):
					if cmd[2] == '1' or cmd[2] == '2':
						obj.order = int(cmd[2])
						if not obj.me == None:
							obj.initfem()
						print(f"FEM order set to '{cmd[2]}'")
					else:
						print("Invalid order. Should be eithr '1' or '2'")

			elif(cmd[0] == 'simulate'):
				obj.fem.setMaterial(obj.matmap)
				obj.fem.freq = obj.freq

				obj.fem.calculateElementalMatrices()
				ns = np.size(obj.freq)
				obj.trans = np.zeros((ns,1),dtype = complex)
				obj.ref = np.zeros((ns,1),dtype = complex)
				obj.fem.assemble_system()
				obj.fem.setPorts(obj.ports)
				tr = []
				ref = []
				for i in range(ns):
					obj.trans[i],obj.ref[i],t = obj.fem.solve_numeric_system(obj.freq[i])

					printprogress(i,ns-1)
				obj.data['s11'] = [obj.freq,np.abs(obj.ref)]
				obj.data['s21'] = [obj.freq,np.abs(obj.trans)]
				obj.data['s11db'] = [obj.freq,20*log10(np.abs(obj.ref))]
				obj.data['s21db'] = [obj.freq,20*log10(np.abs(obj.trans))]

			elif(cmd[0] == 'addplot'):
				if len(cmd) == 2:
					if cmd[1] in obj.data.keys():
						obj.plotlist.append(cmd[1].strip())
					else:
						print(f"'{cmd[1]}' not in data")
			elif(cmd[0] == 'rmplot'):
				if len(cmd) == 2:
					obj.plotlist.remove(cmd[1])
			elif(cmd[0] == 'loaddata'):
				if len(cmd) == 3:
					with open(cmd[1],'r') as file:
						d = file.read()
					x = d.split('\n')
					res = []
					fr = []
					for i in x:

						if i.strip() == '' or i.startswith('%'):
							continue
						y = i.split()
						#print(y)
						fr.append(float(y[0].strip()))
						res.append(float(y[1].strip()))
					obj.data[cmd[2]] = [fr,res]
			elif(cmd[0] == 'savedata'):
				if len(cmd) == 3:
					with open(cmd[1],'w') as file:
						for i in zip(obj.data[cmd[2]][0],obj.data[cmd[2]][1]):
							file.write(f"{i[0]} {i[1][0]}\n")

					



						
			elif(cmd[0] == 'plot'):
				for curve in obj.plotlist:
					'''
					if(curve == 's21'):
						if(obj.plttype == 'db'):
							plt.plot(obj.freq,20*log10(np.abs(obj.trans)))
						elif(obj.plttype == 'lin'):
							plt.plot(obj.freq,np.abs(obj.trans))
					elif(curve == 's11'):
						if(obj.plttype == 'db'):
							plt.plot(obj.freq,20*log10(np.abs(obj.ref)))
						elif(obj.plttype == 'lin'):
							plt.plot(obj.freq,np.abs(obj.ref))
					'''
					plt.plot(obj.data[curve][0],obj.data[curve][1])
				if(obj.grid):
					plt.grid()
				plt.legend(obj.plotlist)
				plt.show()
						

							
					
			elif(cmd[0] == 'execute' or cmd[0] == 'run'):
				if(not cmd[1].endswith('.fem')):
					cmd[1] = cmd[1] + '.fem'
				with open(cmd[1],'r') as f:
					code = f.read().strip()
				code = code.split('\n')
				for line in code:
					obj.process(line)


			elif(cmd[0] == 'reload'):
				
				reload(FEM)
				print('Reloade FEM Verison ', FEM.FEM.version())
				reload(MeshEngine)
			else:
				print("'", cmd[0],"' is not recognized as a command", sep ='')
	def initfem(obj):
		obj.fem = FEM.FEM(obj.order,obj.me.Mesh)
		obj.fem.generateDof()
		print("Number of DOFS: ", obj.fem.DOF.maxdof)
		obj.fem.setPorts(obj.ports)
		obj.fem.setExcitation(obj.exc_port)
		obj.state = 'Initialized with ' + obj.fname

if __name__ == "__main__":
	fi = fem_interactive()
	while(True):
		cmd = input(">>")
		fi.process(cmd)