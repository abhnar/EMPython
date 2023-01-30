from scipy.io import loadmat as load
from types import SimpleNamespace
from numpy import max as max
from numpy import shape as size
from numpy import unique 
from numpy import max
from numpy import min
from numpy import zeros
from numpy import ones
from numpy import where as find
from numpy.linalg import norm
from Nedelec import Nedelec
from math import sin,cos,pi
import inspect  
import numpy as np
from Utils import printprogress
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from math import sqrt
import time
from Utils import spsolveinterface,delete_from_csr
from Utils import dprint
from mkllibrary import MKL_sparse

class FEM:
	version = 1.0
	mu0=4*pi*1e-7;
	eps0=8.85418781762039e-12;
	ports = [];
	excitaion = 0;
	order = 0
	DOF = SimpleNamespace(dofmap=None,tridofmap =None, pecdof = None, maxdof=-1, automatic=True)
	MATS = SimpleNamespace(P=None,C =None, D = None, automatic=True)
	SYSTEM = SimpleNamespace(E=None,F =None, P = None,C =None, D = None, automatic=True)
	TIQUAD = SimpleNamespace(weights = [0.225, 0.13239415278851, 0.13239415278851, 0.13239415278851, 0.12593918054483, 0.12593918054483, 0.12593918054483],
							 xi = [0.3333,	0.4701,	0.4701,	0.0597,	0.1013,	0.1013,	0.7974],
							 eta = [0.3333,	0.4701,	0.0597,	0.4701,	0.1013,	0.7974,	0.1013], Nq = 7, automatic=True)
	
	ELEMENTS = SimpleNamespace(E_e=None, F_e =None, B_e = None,epsilonr=None,mur =None, sigma = None,automatic=True)
	WG = SimpleNamespace(automatic=True)
	SHAPE = SimpleNamespace(Wij=None,CWij =None,Wij_m=None,CWij_m =None, Nq = 0, W = None, Ndof =0, automatic=True)
	MATLAB = {}
	TIME = SimpleNamespace(automatic=True)

	

	def __init__(self,order,Mesh):
		self.Mesh = Mesh;
		##print('Defining finite element of order ',order, ' with Mesh ', Mesh["filename"]);
		self.order = order
		self.SHAPE.Wij, self.SHAPE.CWij, self.SHAPE.W, self.SHAPE.Nq , self.SHAPE.Ndof = Nedelec.getW(order,orientation = 1)
		self.SHAPE.Wij_m, self.SHAPE.CWij_m, self.SHAPE.W, self.SHAPE.Nq , self.SHAPE.Ndof = Nedelec.getW(order,orientation = -1)
		self.ELEMENTS.epsilonr = ones(self.Mesh["ntet"],dtype=np.float)
		self.ELEMENTS.mur = ones(self.Mesh["ntet"],dtype=np.float)
		self.ELEMENTS.sigma = ones(self.Mesh["ntet"],dtype=np.float)
		self.mklsp = MKL_sparse()


	def generateDof(self):
		if self.order == 1:
			self.DOF.dofmap = self.Mesh["tet2edges"]
			self.DOF.tridofmap = self.Mesh["tri2edges"]
			self.DOF.maxdof = max(self.DOF.dofmap)+1

			#tmp = load('dof.mat')
			#self.DOF.dofmap = tmp["dofmap"] - 1
			#self.DOF.tridofmap = tmp["tridofmap"] - 1

			
		else:
			#tmp = load('dof.mat')
			ne = self.Mesh["nedges"]
			nf = self.Mesh["nfaces"]

			edges = self.Mesh["tet2edges"]
			
			edges_q = np.empty((edges.shape[0],12), dtype=edges.dtype)
			for i in range(6):
				ix1 = find(self.Mesh["edges_dir_n"] [:,i]==1)[0]
				ix2 = find(self.Mesh["edges_dir_n"] [:,i]==0)[0]
				##print(size(ix1),size(ix2))
				edges_q[ix1,i*2] = edges[ix1,i]*2
				edges_q[ix1,i*2+1] = edges[ix1,i]*2+1

				edges_q[ix2,i*2] = edges[ix2,i]*2+1
				edges_q[ix2,i*2+1] = edges[ix2,i]*2



			
			edges = edges_q;

			faces1 = self.Mesh["tet2faces"] + 2*ne
			faces2 = self.Mesh["tet2faces"] + 2*ne + nf

			self.DOF.dofmap = np.column_stack((edges,faces1,faces2))


			tried1 =self.Mesh["tri2edges"];
			tried2 = self.Mesh["tri2edges"]*2+1

			tried = np.empty((tried1.shape[0],6), dtype=tried1.dtype)
			tried[:,0::2] = tried1*2
			tried[:,1::2] = tried1*2+1

			trifc1 = self.Mesh["tri2faces"]  + max(edges) + 1
			trifc2 = self.Mesh["tri2faces"]  + max(faces1) + 1
			self.DOF.tridofmap  = np.column_stack((tried,trifc1,trifc2))

		pecdof = self.DOF.tridofmap[self.Mesh["bndry_indx"]["PEC"],:].flatten()
		self.DOF.pecdof = unique(pecdof)
		self.DOF.maxdof =  max(self.DOF.dofmap)+1


			
	def setPorts(self,pset,field=None,type='wg'):
		self.ports = pset
		# This is defined for ports parallel to XY,XZ or YZ exciting only 1st mode
		weights = self.TIQUAD.weights
		xi = self.TIQUAD.xi
		eta = self.TIQUAD.eta
		Nq = self.TIQUAD.Nq
		refnodes = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
		P= {}
		C= {}
		D= {}
		wga_set = {}
		for p in pset:
			##print(p)
			if not (p in self.Mesh["bndry_indx"].keys() ):
				#print("CRITICAL ERROR: Port surface",p, "is not defined")
				exit()
			portfaces = self.Mesh["bndry_indx"][p]
			portelems = self.Mesh["tetoftri"][p]

			if type == "wg":
				#Evaluate normal 
				delx = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,:].flatten(),0]
				xmin =min(delx);
				delx = max(delx)-min(delx)
				
				
				dely = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,:].flatten(),1]
				ymin = min(dely)
				dely = max(dely)-min(dely)
				

				delz = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,:].flatten(),2]
				zmin = min(delz)
				delz = max(delz)-min(delz)
				
				tmp = np.array([delx,dely,delz]);
				wg_a = max(tmp)

				if(np.round(delz/wg_a,10) == 0):
					##print("Normal In Z Axis")
					if(wg_a == delx):
						##print("'a' in X Axis")
						Phi_1 = lambda x,y,z: [0 , sin((pi*(x-xmin))/delx), 0]
					else:
						##print("'a' in Y Axis")
						Phi_1 = lambda x,y,z: [sin((pi*(y-ymin))/dely) , 0, 0]
				
				elif(np.round(dely/wg_a,10) == 0):
					##print("Normal In Y Axis")
					if(wg_a == delx):
						#print("'a' in X Axis")
						Phi_1 = lambda x,y,z: [0 , 0, sin((pi*(x-xmin))/delx)]
					else:
						##print("'a' in Z Axis")
						Phi_1 = lambda x,y,z: [sin((pi*(z-zmin))/delz),0,0]

				elif(np.round(delx/wg_a,10) == 0):
					##print("Normal In X Axis")
					if(wg_a == delz):
						##print("'a' in Z Axis")
						Phi_1 = lambda x,y,z: [0,sin((pi*(z-zmin))/delz), 0]
					else:
						##print("'a' in Y Axis")
						Phi_1 = lambda x,y,z: [0,0,sin((pi*(y-ymin))/dely)]


				##print(inspect.getsourcelines(Phi_1)[0][0])
				ref_vert = zeros(3,dtype=np.int)
				
				ntri = size(portfaces)[0]
				
				p1 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,0],:]
				p2 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,1],:]
				p3 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,2],:]
				n_vec = np.cross(p2-p1,p3-p1);
				area = 1/2*norm(n_vec,axis=1);
				#n_hats =  n_vec/(area[:,None]*2)

				

				P_vec = zeros(self.DOF.maxdof,dtype=np.float)
				C_vec = zeros(self.DOF.maxdof,dtype=np.float)
				dval = 0
				for ele in range(ntri):
					tet_ix = portelems[ele]
					tri_ix = portfaces[ele]

					ref_vert[0]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,0])[0];
					ref_vert[1]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,1])[0];
					ref_vert[2]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,2])[0];


					ntridof = size(self.DOF.tridofmap)[1];
					bedgeoftet = zeros(ntridof,dtype = np.int)

					for i in range(ntridof):
							bedgeoftet[i]=find(self.DOF.dofmap[tet_ix,:]==self.DOF.tridofmap[tri_ix,i])[0]
					

					SP=refnodes[ref_vert,:]
					AB=SP[1,:]-SP[0,:];
					AC=SP[2,:]-SP[0,:];
					n_hat_ref=np.cross(AB,AC)/norm(np.cross(AB,AC));

					Pn=np.setdiff1d([0,1,2,3],ref_vert);
					AD=refnodes[Pn,:]-SP[0,:];
					
					n_hat_ref = n_hat_ref*-np.sign(np.inner(n_hat_ref,AD[0]));
					

					xyz = refnodes[ref_vert,:]
					X= zeros((Nq,3),dtype=np.float)
					for w in range(Nq):
						X[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]

					xyz = self.Mesh["nodes"][self.Mesh["tri2nodes"][tri_ix,:],:]
					Y = zeros((Nq,3),dtype=np.float)
					for w in range(Nq):
						Y[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]
					Wij,CWij,Nq = Nedelec.getW(self.order,X,self.Mesh["tetorient"][tet_ix])


					for i in range(ntridof):
						local_edge_ix = bedgeoftet[i]
						global_edge_ix = self.DOF.dofmap[tet_ix,bedgeoftet[i]]
						int_val = 0;
						for w in range(Nq):
							I1 = Wij[w,:,local_edge_ix]
							I2 = Phi_1(Y[w,0],Y[w,1],Y[w,2])
							
							int_val = int_val + weights[w]*np.dot(np.matmul(self.Mesh["J_IT"][:,:,tet_ix],I1),I2)*area[ele]
							
						P_vec[global_edge_ix] = P_vec[global_edge_ix] + int_val

						int_val = 0;
						for w in range(Nq):
							I1 = np.cross(np.cross(n_hat_ref, Wij[w,:,local_edge_ix]),n_hat_ref)
							I2 = Phi_1(Y[w,0],Y[w,1],Y[w,2])
							
							int_val = int_val + weights[w]*np.dot(np.matmul(self.Mesh["J_IT"][:,:,tet_ix],I1),I2)*area[ele]
							
						C_vec[global_edge_ix] = C_vec[global_edge_ix] + int_val
					
					for w in range(Nq):
						I1 = Phi_1(Y[w,0],Y[w,1],Y[w,2])
						dval = dval + weights[w]*np.dot(I1,I1)*area[ele]
			if type == "num":
				wg_a = 1
				Phi = field[p][1]
				Kz10 = field[p][0]
				#Evaluate normal 
				
				ref_vert = zeros(3,dtype=np.int)
				
				ntri = size(portfaces)[0]
				
				p1 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,0],:]
				p2 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,1],:]
				p3 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,2],:]
				n_vec = np.cross(p2-p1,p3-p1);
				area = 1/2*norm(n_vec,axis=1);
				#n_hats =  n_vec/(area[:,None]*2)

				

				P_vec = zeros(self.DOF.maxdof,dtype=np.float)
				C_vec = zeros(self.DOF.maxdof,dtype=np.float)
				dval = 0
				for ele in range(ntri):
					tet_ix = portelems[ele]
					tri_ix = portfaces[ele]

					ref_vert[0]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,0])[0];
					ref_vert[1]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,1])[0];
					ref_vert[2]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,2])[0];


					ntridof = size(self.DOF.tridofmap)[1];
					bedgeoftet = zeros(ntridof,dtype = np.int)

					for i in range(ntridof):
							bedgeoftet[i]=find(self.DOF.dofmap[tet_ix,:]==self.DOF.tridofmap[tri_ix,i])[0]
					

					SP=refnodes[ref_vert,:]
					AB=SP[1,:]-SP[0,:];
					AC=SP[2,:]-SP[0,:];
					n_hat_ref=np.cross(AB,AC)/norm(np.cross(AB,AC));

					Pn=np.setdiff1d([0,1,2,3],ref_vert);
					AD=refnodes[Pn,:]-SP[0,:];
					
					n_hat_ref = n_hat_ref*-np.sign(np.inner(n_hat_ref,AD[0]));
					

					xyz = refnodes[ref_vert,:]
					X= zeros((Nq,3),dtype=np.float)
					for w in range(Nq):
						X[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]

					xyz = self.Mesh["nodes"][self.Mesh["tri2nodes"][tri_ix,:],:]
					Y = zeros((Nq,3),dtype=np.float)
					for w in range(Nq):
						Y[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]
					Wij,CWij,Nq = Nedelec.getW(self.order,X,self.Mesh["tetorient"][tet_ix])


					for i in range(ntridof):
						local_edge_ix = bedgeoftet[i]
						global_edge_ix = self.DOF.dofmap[tet_ix,bedgeoftet[i]]
						int_val = 0;
						for w in range(Nq):
							I1 = Wij[w,:,local_edge_ix]
							I2 = [Phi[ele][0][w],Phi[ele][1][w],0]
							
							int_val = int_val + weights[w]*np.dot(np.matmul(self.Mesh["J_IT"][:,:,tet_ix],I1),I2)*area[ele]
							
						P_vec[global_edge_ix] = P_vec[global_edge_ix] + int_val

						int_val = 0;
						for w in range(Nq):
							I1 = np.cross(np.cross(n_hat_ref, Wij[w,:,local_edge_ix]),n_hat_ref)
							I2 = [Phi[ele][0][w],Phi[ele][1][w],0]
							
							int_val = int_val + weights[w]*np.dot(np.matmul(self.Mesh["J_IT"][:,:,tet_ix],I1),I2)*area[ele]
							
						C_vec[global_edge_ix] = C_vec[global_edge_ix] + int_val
					
					for w in range(Nq):
						I1 = [Phi[ele][0][w],Phi[ele][1][w],0]
						dval = dval + weights[w]*np.dot(I1,I1)*area[ele]
			#print("P_vec max in setPort",np.max(P_vec))
			P[p] = P_vec
			C[p] = C_vec
			D[p] = dval
			wga_set[p] = wg_a

		self.WG.wg_a = wg_a;	
		self.MATS.P = P;
		self.MATS.C = C;
		self.MATS.D = D;
		self.MATS.ports = pset
		self.WG.wga_set = wga_set


	def setExcitation(self,pset):
		A={}
		for p in pset:
			A[p] = 1
		self.MATS.A = A
	
	def setMaterial(self,matmap):
		for key in matmap:
			if not (key in self.Mesh["domain_indx"].keys() ):
				##print("CRITICAL ERROR: Domain '"+key+"' is not defined")
				#exit()
				continue
			self.ELEMENTS.epsilonr[self.Mesh["domain_indx"][key]] = matmap[key][0]
			self.ELEMENTS.mur[self.Mesh["domain_indx"][key]] = matmap[key][1]
			self.ELEMENTS.sigma[self.Mesh["domain_indx"][key]] = matmap[key][2]
		
	def calculateElementalMatrices(self):
		print('Calculating Elemental Matrices')
		J = self.Mesh["J"].transpose(2,0,1)
		J_IT = self.Mesh["J_IT"].transpose(2,0,1)
		J_Det = self.Mesh["J_Det"]

		self.ELEMENTS.E = zeros((self.SHAPE.Ndof,self.SHAPE.Ndof,self.Mesh["ntet"]))
		self.ELEMENTS.F = zeros((self.SHAPE.Ndof,self.SHAPE.Ndof,self.Mesh["ntet"]))
		ix = find(self.Mesh["tetorient"]==1)[0]
		ix_m = find(self.Mesh["tetorient"]==-1)[0]
		
		cN = [ ([None] * self.SHAPE.Ndof) for row in range(self.SHAPE.Nq) ]
		cN_m = [ ([None] * self.SHAPE.Ndof) for row in range(self.SHAPE.Nq) ]
		N = [ ([None] * self.SHAPE.Ndof) for row in range(self.SHAPE.Nq) ]
		N_m = [ ([None] * self.SHAPE.Ndof) for row in range(self.SHAPE.Nq) ]
		
		##print(self.SHAPE.Wij[0,:,0])
		#print('Evaluating Local Matrices')
		for p in range(self.SHAPE.Nq):
			for i in range(self.SHAPE.Ndof):
				cN[p][i] =   (J[ix,:,None]*self.SHAPE.CWij[p,:,i]).sum(axis=3)[:,:,0]#self.PiolaTransform(J[ix,:,:],self.SHAPE.CWij[p,:,i])
				cN_m[p][i] = (J[ix_m,:,None]*self.SHAPE.CWij_m[p,:,i]).sum(axis=3)[:,:,0]#self.PiolaTransform(J[ix_m,:,:],self.SHAPE.CWij_m[p,:,i])
				
				N[p][i] =   (J_IT[ix,:,None]*self.SHAPE.Wij[p,:,i]).sum(axis=3)[:,:,0]#self.PiolaTransform(J_IT[ix,:,:],self.SHAPE.CWij[p,:,i])
				N_m[p][i] =  (J_IT[ix_m,:,None]*self.SHAPE.Wij_m[p,:,i]).sum(axis=3)[:,:,0]#self.PiolaTransform(J_IT[ix_m,:,:],self.SHAPE.CWij_m[p,:,i])

		##print(self.SHAPE.Wij[0,:,0])
		#p=0
		#i=0

		##print((J_IT[ix,:,None]*self.SHAPE.Wij[p,:,i]).sum(axis=3)[:,:,0])
		##print(N[p][i])

		for p in range(self.SHAPE.Nq):
			for i in range(self.SHAPE.Ndof):
				for j in range(i,self.SHAPE.Ndof):
					self.ELEMENTS.E[i,j,ix] = self.ELEMENTS.E[i,j,ix] + self.SHAPE.W[p]*(cN[p][i]*cN[p][j]).sum(axis = 1)
					self.ELEMENTS.E[i,j,ix_m] = self.ELEMENTS.E[i,j,ix_m] + self.SHAPE.W[p]*(cN_m[p][i]*cN_m[p][j]).sum(axis = 1)

					self.ELEMENTS.F[i,j,ix] = self.ELEMENTS.F[i,j,ix] + self.SHAPE.W[p]*(N[p][i]*N[p][j]).sum(axis = 1)
					self.ELEMENTS.F[i,j,ix_m] = self.ELEMENTS.F[i,j,ix_m] + self.SHAPE.W[p]*(N_m[p][i]*N_m[p][j]).sum(axis = 1)
			printprogress(p,self.SHAPE.Nq-1)
		##print((self.ELEMENTS.E[:,:,:]*(1/J_Det))	)

		self.ELEMENTS.E  = self.UT2SYM3D(self.ELEMENTS.E )
		self.ELEMENTS.F  = self.UT2SYM3D(self.ELEMENTS.F )
		self.ELEMENTS.E = self.ELEMENTS.E[:,:,:]*(1/J_Det)
		self.ELEMENTS.F = self.ELEMENTS.F[:,:,:]*(J_Det)
	
	def assemble(self):
		print("Assembling elemental matrices (E,F)")
		NTet=self.Mesh['ntet'];
		self.ELEMENTS.E = self.ELEMENTS.E[:,:,:]*(1/self.ELEMENTS.mur)
		#self.ELEMENTS.F = self.ELEMENTS.F[:,:,:]*(self.ELEMENTS.epsilonr)
		for i in range(NTet):
			self.ELEMENTS.F[:,:,i] = self.ELEMENTS.F[:,:,i]*self.ELEMENTS.epsilonr[i]


		
		tmp=np.tile(np.transpose(self.DOF.dofmap),(self.SHAPE.Ndof,1))
		X=np.reshape(tmp,(self.SHAPE.Ndof,self.SHAPE.Ndof,NTet));
		

		Y=np.transpose(X,(1,0,2))
		X=X.flatten('F')
		##print(X[1:10])
		Y=Y.flatten('F')
		
		self.MATS.E = sps.csr_matrix((self.ELEMENTS.E.flatten('F'),(X,Y)),dtype=complex);
		self.MATS.F = sps.csr_matrix((self.ELEMENTS.F.flatten('F'),(X,Y)),dtype=complex);

	def assemble_system(self):
		self.assemble()
		print("Assembling the system (A,D,P,C, IH, IE)")
		A = np.array([[1],[0]])
		pecedge = self.DOF.pecdof
		##print(pecedge)
		##print("Original Size ", size(self.MATS.E))
		##print("pecdof Size ", size(pecedge))
		
		portmap = {}
		cnt = 0
		for key in self.MATS.ports:
			portmap[key] = cnt
			cnt = cnt+1
		Dlist =[0]*cnt
		for key in self.MATS.D:
			Dlist[portmap[key]]=self.MATS.D[key]
		D = np.diag(Dlist)
		
		
		Plist =[None]*cnt
		for key in self.MATS.P:
			Plist[portmap[key]] = self.MATS.P[key]

		Clist =[None]*cnt
		for key in self.MATS.P:
			Clist[portmap[key]] = self.MATS.P[key]
		
		IE = -np.matmul(D,A)
		
		

		C = np.row_stack(Clist)
		C = np.delete(C, (pecedge), 1)
		
		
		P = np.column_stack(Plist)
		P = np.delete(P, (pecedge), 0)

		IH = np.matmul(P,A)
		##print(size(IH))
		

		
		#K = np.row_stack(np.column_stack ((D,C)),sps.column_stack ((P,self.MATS.E)))
		D = sps.csr_matrix(D,dtype=complex)
		C = sps.csr_matrix(C,dtype=complex)

		E = delete_from_csr(self.MATS.E, row_indices=pecedge)
		E = delete_from_csr(E, col_indices=pecedge)

		F = delete_from_csr(self.MATS.F, row_indices=pecedge)
		F = delete_from_csr(F, col_indices=pecedge)


		n = size(E)[0]

		b = np.row_stack((IE,IH))
		##print(size(b),n)

		#C  = delete_from_csr(C, col_indices=pecedge)
		#P = delete_from_csr(P, row_indices=pecedge)


		Z = sps.csr_matrix((n, n), dtype = np.float)
		#K = sps.vstack((sps.hstack ((D,C)),sps.hstack ((P,self.MATS.E))) ) 

		self.SYSTEM.P = sps.vstack((sps.hstack ((D*0,C*0)),sps.hstack ((P,Z))) ) 
		self.SYSTEM.C = sps.vstack((sps.hstack ((D*0,C)),sps.hstack ((P*0,Z))) ) 
		##print(size(P_))
		##print(size(C_))
		self.SYSTEM.E =  sps.vstack((sps.hstack ((D,C*0)),sps.hstack ((P*0,E))) ) 
		self.SYSTEM.F =  sps.vstack((sps.hstack ((D*0,C*0)),sps.hstack ((P*0,F))) ) 
		self.SYSTEM.size = n
		self.SYSTEM.b = b

		
		self.MATLAB = {"E":E,"F":F,"P":P,"C":C,"D":D,"b":b}

	def solve_numeric_system_multimode(self,field,f):
		self.mklsp = MKL_sparse();
		
		eta0 = 120*pi;
		pecedge = self.DOF.pecdof
		K0=2*pi*f*1e9*sqrt(self.mu0*self.eps0);
		nports= len(field)
		nports = 0
		port_indices = {}
		cnt = 0
		Phiset = []
		for key in field.keys():
			port_indices[key] = np.arange(nports,nports+field[key][2].nmodes)
			nports = field[key][2].nmodes + nports
			for m in field[key][2].Modes:
				Phiset.append(m)
				##print(m)


		P = np.zeros((self.DOF.maxdof,nports),dtype = np.complex)
		C = np.zeros((nports,self.DOF.maxdof),dtype = np.complex)
		D = np.zeros((nports,nports),dtype = np.complex)
		weights = self.TIQUAD.weights
		xi = self.TIQUAD.xi
		eta = self.TIQUAD.eta
		Nq = self.TIQUAD.Nq
		refnodes = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
		cnt = 0
		for key in field.keys():
			p = key
			portfaces = self.Mesh["bndry_indx"][p]
			portelems = self.Mesh["tetoftri"][p]
			#p_cost[key] =(-i*self.WG.K0*eta0)*(i*self.WG.Kz10[key]);
			Modes = field[key][2].Modes
			Kz10s = field[key][2].Kz10s
			dval = 0
			for m in range(field[key][2].nmodes):
				Phi = Modes[m]
				Kz10 = Kz10s[m]
				#print("Kz10",Kz10)
				#Evaluate normal 
				
				ref_vert = zeros(3,dtype=np.int)
				
				ntri = size(portfaces)[0]
				
				p1 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,0],:]
				p2 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,1],:]
				p3 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,2],:]
				n_vec = np.cross(p2-p1,p3-p1);
				area = 1/2*norm(n_vec,axis=1);
				#n_hats =  n_vec/(area[:,None]*2)

				

				P_vec = zeros(self.DOF.maxdof,dtype=np.complex)
				C_vec = zeros(self.DOF.maxdof,dtype=np.complex)
				dval = 0
				for ele in range(ntri):
					tet_ix = portelems[ele]
					tri_ix = portfaces[ele]

					ref_vert[0]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,0])[0];
					ref_vert[1]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,1])[0];
					ref_vert[2]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,2])[0];


					ntridof = size(self.DOF.tridofmap)[1];
					bedgeoftet = zeros(ntridof,dtype = np.int)

					for i in range(ntridof):
							bedgeoftet[i]=find(self.DOF.dofmap[tet_ix,:]==self.DOF.tridofmap[tri_ix,i])[0]
					

					SP=refnodes[ref_vert,:]
					AB=SP[1,:]-SP[0,:];
					AC=SP[2,:]-SP[0,:];
					n_hat_ref=np.cross(AB,AC)/norm(np.cross(AB,AC));

					Pn=np.setdiff1d([0,1,2,3],ref_vert);
					AD=refnodes[Pn,:]-SP[0,:];
					
					n_hat_ref = n_hat_ref*-np.sign(np.inner(n_hat_ref,AD[0]));
					

					xyz = refnodes[ref_vert,:]
					X= zeros((Nq,3),dtype=np.float)
					for w in range(Nq):
						X[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]

					xyz = self.Mesh["nodes"][self.Mesh["tri2nodes"][tri_ix,:],:]
					Y = zeros((Nq,3),dtype=np.float)
					for w in range(Nq):
						Y[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]
					Wij,CWij,Nq = Nedelec.getW(self.order,X,self.Mesh["tetorient"][tet_ix])


					for i in range(ntridof):
						local_edge_ix = bedgeoftet[i]
						global_edge_ix = self.DOF.dofmap[tet_ix,bedgeoftet[i]]
						int_val = 0;
						for w in range(Nq):
							I1 = Wij[w,:,local_edge_ix]
							I2 = [Phi[ele][0][w],Phi[ele][1][w],0]
							
							int_val = int_val + weights[w]*np.dot(np.matmul(self.Mesh["J_IT"][:,:,tet_ix],I1),I2)*area[ele]
							
						P_vec[global_edge_ix] = P_vec[global_edge_ix] + int_val

						int_val = 0;
						for w in range(Nq):
							I1 = np.cross(np.cross(n_hat_ref, Wij[w,:,local_edge_ix]),n_hat_ref)
							I2 = [Phi[ele][0][w],Phi[ele][1][w],0]
							
							int_val = int_val + weights[w]*np.dot(np.matmul(self.Mesh["J_IT"][:,:,tet_ix],I1),I2)*area[ele]
							
						C_vec[global_edge_ix] = C_vec[global_edge_ix] + int_val

					for w in range(Nq):
						I1 = [Phi[ele][0][w],Phi[ele][1][w],0]
						dval = dval + weights[w]*np.dot(I1,I1)*area[ele]
					
					
					
						
						
				
				#print("\ndval\n",dval)
				i = complex(0,1)
				P[:,cnt] = P_vec*(-i*K0*eta0)*(i*Kz10)
				C[cnt,:] = C_vec*(1/(i*K0*eta0))
				#D[cnt,cnt] = dval
				cnt = cnt+1
		
		
				
		
		
		for key in port_indices.keys():
			pix = port_indices[key]
			portfaces = self.Mesh["bndry_indx"][key]
			portelems = self.Mesh["tetoftri"][key]
			p1 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,0],:]
			p2 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,1],:]
			p3 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,2],:]
			n_vec = np.cross(p2-p1,p3-p1);
			area = 1/2*norm(n_vec,axis=1);
			ntri = size(portfaces)[0]
			cnt = 0
			Modes = field[key][2].Modes
			for i in range(field[key][2].nmodes):
				
				Phi1 = Modes[i]
				for j in range(field[key][2].nmodes):
					
					Phi2 = Modes[j]
					dval = 0
					
					for ele in range(ntri):
						for w in range(Nq):
							I1 = [Phi1[ele][0][w],Phi1[ele][1][w],0]
							I2 = [Phi2[ele][0][w],Phi2[ele][1][w],0]
							dval = dval + weights[w]*np.dot(I1,I2)*area[ele]
					D[pix[i],pix[j]] = dval
					cnt=cnt+1
					
					
					
					


		#breakpoint()
			
		P = np.delete(P, (pecedge), 0)
		C = np.delete(C, (pecedge), 1)
		A = np.zeros((nports,1), dtype = np.complex)
		A[0,0] = 1


		IE = -np.matmul(D,A)
		IH = np.matmul(P,A)
		b = np.row_stack((IE,IH))
		b = b.astype(complex)
		##print(A,D,IE)

		P = sps.csr_matrix(P)
		C = sps.csr_matrix(C)
		D = sps.csr_matrix(D)
		##print(P.shape,C.shape,D.shape)

		##print(P.getformat(),C.getformat(),D.getformat())

		E = delete_from_csr(self.MATS.E, row_indices=pecedge)
		E = delete_from_csr(E, col_indices=pecedge)

		F = delete_from_csr(self.MATS.F, row_indices=pecedge)
		F = delete_from_csr(F, col_indices=pecedge)

		#P =  delete_from_csr(P, row_indices=pecedge)
		#C =  delete_from_csr(C, col_indices=pecedge)

		K = E - K0**2*F
		nn = K.shape[0]
		K =  sps.vstack((sps.hstack ((D,C)),sps.hstack ((P,K))) ) 
		K = sps.csr_matrix(K,dtype=complex)
		##print(K.getformat(),nports)
		
		##print("max Ie, max Ih, max D",np.max(IE), np.max(IH), np.max(D))
		##print(IE)
		##print("max P",np.max(P))
		##print("max Ie, max Ih, max D",np.max(IE), np.max(IH), np.max(D))
		
		

		
		#x= spsolveinterface(K,b,13)
		if(self.mklsp.status == 1):
			x = self.mklsp.solve(K,b,mattype = 13)
		else:
			x=sps.linalg.spsolve(K,b)

		##print('Time : ',time.time()-start_time)
		
		t = x[0:nports]
		#print(port_indices.keys())
		#print("S21:(", port_indices['OP'],")", np.sum(x[port_indices['OP']]))
		#print("S11:(", port_indices['IP'],")", np.sum(x[port_indices['IP']]))
		s11 = 0;
		s21 = 0
		for i in port_indices['IP']:
			s11 = s11+np.abs(x[i])**2
		s11 = np.sqrt(s11)
		for i in port_indices['OP']:
			s21 = s21+np.abs(x[i])**2
		s21 = np.sqrt(s21)
		#print("power :", s11**2+s21**2)
		self.ippow = D[0,0]
		y = x[-nn:]
		
		self.Ef = np.zeros((self.DOF.maxdof,1),dtype= np.complex)
		pecedge = self.DOF.pecdof
		ins = np.arange(0, self.DOF.maxdof)
		ins[pecedge] = -1
		ins = np.where(ins != -1)[0]
		self.Ef[ins,0] = y.flatten()
		self.oppow = self.calcTrans()
		#dprint((self.ippow,self.oppow))
		#s21 = np.sqrt(self.oppow/self.ippow)
		#s11 = np.sqrt(1-s21**2)
		#return np.sum(x[port_indices['Port-2']]),np.sum(x[port_indices['Port-1']]),t
		return s21,s11,t
		#return x[port_indices['Port-2'][0]],x[port_indices['Port-1'][0]],t
		#return 0,0,0

	def solve_numeric_system(self,f):
		
		self.buildWG(f)
		i = complex(0,1)
		eta0 = 120*pi;
		p_cost = {}
		self.WG.K0=2*pi*f*1e9*sqrt(self.mu0*self.eps0);

		for key in self.WG.Kz10.keys():
			p_cost[key] =(-i*self.WG.K0*eta0)*(i*self.WG.Kz10[key]);


		#print("pcost ", p_cost)
		
		c_cost = (1/(i*self.WG.K0*eta0));

		A = np.array([[1],[0]])
		pecedge = self.DOF.pecdof
		##print(pecedge)
		##print("Original Size ", size(self.MATS.E))
		##print("pecdof Size ", size(pecedge))
		
		portmap = {}
		cnt = 0
		for key in self.MATS.ports:
			portmap[key] = cnt
			cnt = cnt+1
		Dlist =[0]*cnt
		for key in self.MATS.D:
			Dlist[portmap[key]]=self.MATS.D[key]
		D = np.diag(Dlist)
		
		
		Plist =[None]*cnt
		
		for key in self.MATS.P:
			Plist[portmap[key]] = self.MATS.P[key]*p_cost[key]
			

		Clist =[None]*cnt
		for key in self.MATS.P:
			Clist[portmap[key]] = self.MATS.P[key]*c_cost
		
		IE = -np.matmul(D,A)
		
		

		C = np.row_stack(Clist)
		C = np.delete(C, (pecedge), 1)
		
		
		P = np.column_stack(Plist)
		P = np.delete(P, (pecedge), 0)

		IH = np.matmul(P,A)
		##print(size(IH))
		
		
		
		#K = np.row_stack(np.column_stack ((D,C)),sps.column_stack ((P,self.MATS.E)))
		D = sps.csr_matrix(D,dtype=complex)
		C = sps.csr_matrix(C,dtype=complex)
		P = sps.csr_matrix(P,dtype=complex)

		E = delete_from_csr(self.MATS.E, row_indices=pecedge)
		E = delete_from_csr(E, col_indices=pecedge)

		F = delete_from_csr(self.MATS.F, row_indices=pecedge)
		F = delete_from_csr(F, col_indices=pecedge)

		K = E - self.WG.K0**2*F
		n = size(E)[0]

		b = np.row_stack((IE,IH))
		##print(size(b),n)

		#C  = delete_from_csr(C, col_indices=pecedge)
		#P = delete_from_csr(P, row_indices=pecedge)
		#print("max P", np.max(P))

		Z = sps.csr_matrix((n, n), dtype = np.float)
		#K = sps.vstack((sps.hstack ((D,C)),sps.hstack ((P,self.MATS.E))) ) 
		nn = K.shape[0]
		#print(K.getformat(),D.getformat(),C.getformat(),P.getformat())
		K =  sps.vstack((sps.hstack ((D,C)),sps.hstack ((P,K))) ) 
		K = sps.csr_matrix(K,dtype=complex)
		##print("max Ie, max Ih, max D",np.max(IE), np.max(IH), np.max(D))
		##print(IE)
		self.SYSTEM.size = n
		b = b.astype(complex)

		start_time = time.time()
		if(self.mklsp.status == 1):
			x = self.mklsp.solve(K,b,mattype = 3)
		else:
			x=sps.linalg.spsolve(K,b)
		##print('Time : ',time.time()-start_time)
		t = time.time()-start_time

		y = x[-nn:]
		
		self.Ef = np.zeros((self.DOF.maxdof,1),dtype= np.complex)
		pecedge = self.DOF.pecdof
		ins = np.arange(0, self.DOF.maxdof)
		ins[pecedge] = -1
		ins = np.where(ins != -1)[0]
		self.Ef[ins,0] = y.flatten()
		self.oppow = self.calcTrans()
		return x[1],x[0],t

	def buildWG(self,f):
		

		self.WG.K0=2*pi*f*1e9*sqrt(self.mu0*self.eps0);

		
		i = complex(0,1)
		
		self.WG.Kz10 = {}
		for p in self.ports:
			a = self.WG.wga_set[p]
			wb=self.WG.K0**2-(pi/a)**2;
			if wb<0:
				wb=-i*sqrt(-wb);
			elif wb>0:
				wb=sqrt(wb);
			self.WG.Kz10[p] = wb
		#print('keys:', self.WG.Kz10.keys(),self.ports)
	def buildWGold(self,f):
		a = self.WG.wg_a
		self.WG.K0=2*pi*f*1e9*sqrt(self.mu0*self.eps0);

		i = complex(0,1)
		wb=self.WG.K0**2-(pi/a)**2;
		if wb<0:
			wb=-i*sqrt(-wb);
		elif wb>0:
			wb=sqrt(wb);
		self.WG.Kz10 = wb
		
		
		
		
		
	def solve(self):

		
		i = complex(0,1)
		#

		eta0 = 120*pi;
		if isinstance(self.WG.Kz10, dict) :
			p_cost = (-i*self.WG.K0*eta0)*(i*self.WG.Kz10['Port-1']);
			c_cost = (1/(i*self.WG.K0*eta0));
		else:
			p_cost = (-i*self.WG.K0*eta0)*(i*self.WG.Kz10);
			c_cost = (1/(i*self.WG.K0*eta0));

		K = self.SYSTEM.E - self.WG.K0**2*self.SYSTEM.F + p_cost*self.SYSTEM.P + c_cost*self.SYSTEM.C
		#b= sps.csr_matrix((size(self.SYSTEM.b)),dtype = complex)
		b = self.SYSTEM.b.astype(complex).copy();
		
		b[2:] = b[2:]*p_cost
		#print(K.getformat())
		#b[2:] = b[2:]*p_cost
		
		#start_time = time.time()
		#x=sps.linalg.spsolve(K,b)
		##print('Time : ',time.time()-start_time)
		self.SYSTEM.K = K
		#self.SYSTEM.b = b
		start_time = time.time()
		x= spsolveinterface(K,b,13)
		##print('Time : ',time.time()-start_time)
		t = time.time()-start_time
		return x[1],x[0],t
		
	def calcTrans(self):
		weights = self.TIQUAD.weights
		xi = self.TIQUAD.xi
		eta = self.TIQUAD.eta
		Nq = self.TIQUAD.Nq
		refnodes = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]])
		portfaces = self.Mesh["bndry_indx"]["OP"]
		portelems = self.Mesh["tetoftri"]["OP"]
		ref_vert = zeros(3,dtype=np.int)
		ntri = size(portfaces)[0]
				
		p1 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,0],:]
		p2 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,1],:]
		p3 = self.Mesh["nodes"][self.Mesh["tri2nodes"][portfaces,2],:]
		n_vec = np.cross(p2-p1,p3-p1);
		area = 1/2*norm(n_vec,axis=1);
		#n_hats =  n_vec/(area[:,None]*2)
		tet2edges = self.DOF.dofmap
		

		int_val = 0
		PLOTDATA = zeros((ntri,4),dtype=complex)
		for ele in range(ntri):
			tet_ix = portelems[ele]
			tri_ix = portfaces[ele]

			ref_vert[0]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,0])[0];
			ref_vert[1]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,1])[0];
			ref_vert[2]= find(self.Mesh["tet2nodes"][tet_ix,:]==self.Mesh["tri2nodes"][tri_ix,2])[0];


			ntridof = size(self.DOF.tridofmap)[1];
			nttetdof = size(self.DOF.dofmap)[1];
			bedgeoftet = zeros(ntridof,dtype = np.int)

			for i in range(ntridof):
					bedgeoftet[i]=find(self.DOF.dofmap[tet_ix,:]==self.DOF.tridofmap[tri_ix,i])[0]
			

			SP=refnodes[ref_vert,:]
			AB=SP[1,:]-SP[0,:];
			AC=SP[2,:]-SP[0,:];
			n_hat_ref=np.cross(AB,AC)/norm(np.cross(AB,AC));

			Pn=np.setdiff1d([0,1,2,3],ref_vert);
			AD=refnodes[Pn,:]-SP[0,:];
			
			n_hat_ref = n_hat_ref*-np.sign(np.inner(n_hat_ref,AD[0]));
			

			xyz = refnodes[ref_vert,:]
			X= zeros((Nq,3),dtype=np.float)
			for w in range(Nq):
				X[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]

			xyz = self.Mesh["nodes"][self.Mesh["tri2nodes"][tri_ix,:],:]
			Y = zeros((Nq,3),dtype=np.float)
			for w in range(Nq):
				Y[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]
			Wij,CWij,Nq = Nedelec.getW(self.order,X,self.Mesh["tetorient"][tet_ix])

			Y = zeros((1,3),dtype=np.float)
			Y[0,0] = 0.5
			Y[0,1] = 0.5
			Y[0,2] = 0
			WW,tmp,tmp2 = Nedelec.getW(self.order,Y,self.Mesh["tetorient"][tet_ix])
			I1 = np.array([0,0,0])
			for j in range(nttetdof):
				I1 = I1 + np.matmul(self.Mesh["J_IT"][:,:,tet_ix],WW[0,:,j])*self.Ef[tet2edges[tet_ix,j]]
			PLOTDATA[ele,0] = np.mean(xyz[:,0])
			PLOTDATA[ele,1] = np.mean(xyz[:,1])
			PLOTDATA[ele,2] = np.real(I1[0])
			PLOTDATA[ele,3] = np.real(I1[1])


			integ = 0
			for w in range(Nq):
				I1 = np.array([0,0,0])
				for j in range(nttetdof):
					I1 = I1 +  np.matmul(self.Mesh["J_IT"][:,:,tet_ix],Wij[w,:,j])*self.Ef[tet2edges[tet_ix,j]]
				#I2 = [Phi[ele][0][w],Phi[ele][1][w],0]
				
				integ = integ + weights[w]*np.dot(I1,I1)*area[ele]
				int_val = int_val + weights[w]*np.dot(I1,I1)*area[ele]
				#dprint(self.Mesh["J_Det"][tet_ix])

				
			


			

				
				
		#from scipy.io import loadmat as load, savemat as save
		#save('pltdata.mat',{'pltdata':PLOTDATA})
		return int_val
					
			
	@staticmethod
	def version():
		return '1.02'
	@staticmethod
	def PiolaTransform(B,N):
		##print(size(N))
		#nelems=size(B)[2];
		#I1 = ones((3,1),dtype = np.int)
		#I2 = ones((nelems,1),dtype = np.int)
		#N2 = N[I1,:,I2];
		#PoT= B* N2;
		#PoT = sum(PoT,axis=1);
		#PoT = (B[:,None,:]*N).sum(axis =1)
		
		PoT = (B[:,:,None]*N).sum(axis=3)[:,:,0]
		##print(PoT)
		##print ('--')
		##print(np.matmul(B[0,:,:],N))
		##print(np.matmul(B[1,:,:],N))
		##print( PoT.shape)
		#PoT = np.einsum('mnr,r->mr', B, N)
		##print(size(PoT))
		return PoT
	@staticmethod
	def UT2SYM3D(E):
		n = size(E)[0]
		return E+ E.transpose([1,0,2]) -  (np.multiply(np.transpose(E,[2,0,1]),np.identity(n))).transpose([1,2,0])










				




		
	 