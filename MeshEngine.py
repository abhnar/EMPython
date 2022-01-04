from Pipeline import Pipeline
from numpy import shape as size
import numpy as np
from numpy import unique as unique
from numpy import sort as sort
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
from Utils import printprogress


class MeshEngine:
	edgeseq =np.array([[0,1],[0,2],[0,3],[1,2],[2,3],[3,1]])
	faceseq =np.array([[1,2,3],[0,3,2],[0,1,3],[0,2,1]])
	Mesh = {}

	def __init__(self):
		pass

	def readComsolFile(self,fname):
	  
		status=0
		pl=Pipeline()
		
		try:
			with open(fname+'.mphtxt') as fp:
				line = fp.readline()
				cnt = 1
				while line and status==0:
					#print("Line {}: {}".format(cnt, line.strip()))
					pl.push(line)
					
					if (line.strip()=='# Mesh point coordinates' or line.strip()=='# Mesh vertex coordinates'):
						status=1
						
						#self.readVertices(fp)
						break
					
					line = fp.readline()
					cnt += 1
				tmp=pl.get(4)
				NN=tmp.split()[0]
				#print("No of Nodes as per file :", NN)
				Nodes=np.zeros((int(NN),3));
				cnt=0
				line = fp.readline()
				#Cordinate reading phase
				while line and status==1:

					if (line.strip()==''):
						status=2
						#print("No of Nodes obtained: ", cnt)
						break
						#self.readVertices(fp)
						#break
					#print(np.fromstring(line.strip(), dtype=float, sep=' '))
					Nodes[cnt,:]=np.fromstring(line.strip(), dtype=float, sep=' ')
					cnt += 1
					line = fp.readline()
				#print(Nodes)
				#print("Size: ",Nodes.shape)
					
				while line and status==2:
					#print("Line {}: {}".format(cnt, line.strip()))
				  
					if (line.strip()=='# Type #2'):
						cnt=0
						status=3
						#self.readVertices(fp)
						#break
					
					line = fp.readline()
					pl.push(line)
					cnt += 1
				
				cnt=0;	
				while line and status==3:
					#print("Line {}: {}".format(cnt, line.strip()))
					
					if (line.strip()=='# Elements'):
						cnt=0
						status=4
						#self.readVertices(fp)
						#break
					
					line = fp.readline()
					pl.push(line)
					cnt += 1
				tmp=pl.get(3)
				Ntri=int(tmp.split()[0])
				#print("No of Triangles as per file :", Ntri)
				Triangles=np.zeros((int(Ntri),3),dtype=np.int);
				Tritype=np.zeros((Ntri,1),dtype=np.int);
				cnt=0;
				#Triangle Reading Phase	
				while line and status==4:
					if (line.strip()==''):
						status=5
						#print("No of Triangles obtained: ", cnt)
						break
						#self.readVertices(fp)
						#break
					#print(np.fromstring(line.strip(), dtype=float, sep=' '))
					Triangles[cnt,:]=np.fromstring(line.strip(), dtype=int, sep=' ')
					cnt += 1
					line = fp.readline()
					pl.push(line)
				#print(Triangles)
				#print("Size: ",Triangles.shape)
					
				line = fp.readline()
				pl.push(line)
				line = fp.readline()
				pl.push(line)
					
				#pl.print()
				line = fp.readline()
				cnt=0
				altcnt=0
				#Triangle Tag Reading Phase	
				while line and status==5:
					if (line.strip()==''):
						status=6
						#print("No of Triangles obtained: ", cnt)
						break
						#self.readVertices(fp)
						#break
					#print(np.fromstring(line.strip(), dtype=float, sep=' '))
					Tritype[cnt,0]=int(line)+1
					if(int(line)==16):
						altcnt=altcnt+1
					cnt += 1
					line = fp.readline()
				#print("# of tritype: ", altcnt)  
				
				for i in range(9):  
					line = fp.readline()
					pl.push(line)
				
				#print("Line {}: {}".format(cnt, line.strip()))
				
				tmp=(pl.get(3))

				Ntet=int(tmp.split()[0])
				
				#print("No of Tetrahedrons as per file :", Ntet)
				Tetrahedrons=np.zeros((int(Ntet),4),dtype=np.int);
				Tettype=np.zeros((Ntet,1),dtype=np.int);
				cnt=0;
				#Reading Tetrahedron
				while line and status==6:
					if (line.strip()==''):
						status=7
					  # print("No of Tetrahedrons obtained: ", cnt)
						break
						#self.readVertices(fp)
						#break
					#print(np.fromstring(line.strip(), dtype=float, sep=' '))
					#print(line)
					Tetrahedrons[cnt,:]=np.fromstring(line.strip(), dtype=int, sep=' ')
					cnt += 1
					line = fp.readline()
					pl.push(line)
				#print(Triangles)
				#print("Size: ",Tetrahedrons.shape)
					
					
				line = fp.readline()
				line = fp.readline()
				line = fp.readline()
				#print("Line {}: {}".format(cnt, line.strip()))
				cnt=0
				while line and status==7:
					if (line.strip()==''):
						status=8
						print("No of Tetytype obtained: ", cnt)
						break
						#self.readVertices(fp)
						#break
					#print(np.fromstring(line.strip(), dtype=float, sep=' '))
					Tettype[cnt,0]=int(line)
					cnt += 1
					line = fp.readline()
					if (not line):
						status=8
						#print("#  Tetytype obtained: ", cnt)
						break
		finally:
			fp.close()
			
			self.Mesh = {'tri2nodes':Triangles,'nodes':Nodes,'tet2nodes':Tetrahedrons,'nn':NN,'ntet':Ntet,'ntri':Ntri,'tettype':Tettype,'tritype':Tritype}

	def readComsolReport(self,fname):
		status=0
		pl=Pipeline()
		BNames=[]
		BTags=[]
		DNames=[]
		DTags=[]
		
		try:
			with open(fname+".html") as fp:
				line = fp.readline()
				while line:
					if('Selections</span>' in line.strip()):
						status=1
						

					if(status==1):

						if('Coordinate Systems</span>' in line.strip()):
							status = 0
							break
						if('<span>Equations</span>' in line.strip()):
							status = 0
							pec=0
							break
						blist=np.array([])
						if('Boundaries' in line.strip()):
							bs=(line.split("</span>",1)[0] )
							bs=(bs.split("Boundaries",1)[1] )
							bs=(bs.replace('â€“','-'))
							bs=bs.split(',')
							for b in bs:
								tmp = b.split('-')
								#print(b,":",len(tmp))
								if len(tmp)>1:
									blist=np.append(blist,np.arange(int(tmp[0]),int(tmp[1])+1))
								else:
									blist=np.append(blist,int(tmp[0]))
							
							tmp=(pl.get(24))
							tmp=(tmp.split("</span>",1)[0] )
							tmp=(tmp.split("<span>",1)[1] )
							BNames.append(tmp)
							BTags.append(blist)
							
							
						if('Boundary' in line.strip()):
							bs=(line.split("</span>",1)[0] )
							bs=(bs.split("Boundary",1)[1] )
							bn=pl.get(24)
							bn=(bn.split("<span>",1)[1] )
							bn=(bn.split("</span>",1)[0] )
							BNames.append(bn)
							BTags.append(np.array([int(bs)]))
							
						if('Domains' in line.strip()):
							bs=(line.split("</span>",1)[0] )
							bs=(bs.split("Domains",1)[1] )
							bs=(bs.replace('â€“','-'))
							bs=bs.split(',')
							for b in bs:
								#print(b)
								tmp = b.split('-')
								if(len(tmp)==2):
									blist=np.append(blist,np.arange(int(tmp[0]),int(tmp[1])+1))
								else:
									blist=np.append(blist,tmp[0])
							
							tmp=(pl.get(24))
							tmp=(tmp.split("</span>",1)[0] )
							tmp=(tmp.split("<span>",1)[1] )
							DNames.append(tmp)
							
							bs = []
							for i in (blist):
								bs.append(int(i))
							DTags.append(bs)
							
							
						elif('Domain' in line.strip()):
							bs=(line.split("</span>",1)[0] )
							bs=(bs.split("Domain",1)[1] )
							bn=pl.get(24)
							bn=(bn.split("<span>",1)[1] )
							bn=(bn.split("</span>",1)[0] )
							DNames.append(bn)
							DTags.append(np.array([int(bs)]))
							
								
							
								
					line = fp.readline()
					pl.push(line.strip())
					
					
			 
		finally:
			fp.close()
			BIds=[0]*(len(BNames))
			DIds=[0]*(len(DNames))
			
			
			
			
			
			self.Mesh['n_bndry'] = len(BNames)
			self.Mesh['bndry_names'] = BNames
			self.Mesh['bndry_tags'] = BTags


			
			self.Mesh['n_domain'] = len(DNames)
			self.Mesh['domain_names'] = DNames
			self.Mesh['domain_tags'] = DTags
	def load(self,Mesh,ports,pecs):
		self.setMeshGeometry(Mesh)
		self.setMeshMap(ports,pecs, Mesh["matmap"])
		
		
	def setMeshGeometry(self, Mesh):
		self.Mesh = Mesh
	def setMeshMap(self, ports,pecs, mmap):
		# Port-1 is IP and Port-2 is OP
		BNames = []
		BIds = []
		BTags = []
		DNames = []
		DIds = []
		DTags = []
		for p in ports:
			BNames.append(p)
			BTags.append(ports[p])
			if p=='Port-1':
				BIds.append(3)

			if p=='Port-2':
				BIds.append(4)
		bdic1 = {}
		BNames.append('PEC')
		BTags.append(pecs)
		BIds.append(2)
		
		matnm = []
		d = {}
		for i in mmap.keys():
			if not mmap[i] in matnm:
				matnm.append(mmap[i])
				d[mmap[i]] = []
			d[mmap[i]].append(i)
		cnt = 0
		for nm in matnm:
			DNames.append(nm)
			DTags.append(d[nm])
			DIds.append(cnt)
		
		
	   
		#tritype= np.array([bdic1[element] for element in tritype])
		
		#self.MeshData["tritype"] = tritype
	   

		self.Mesh['n_bndry'] = len(DNames)
		self.Mesh['bndry_names'] = BNames
		self.Mesh['bndry_tags'] = BTags
		self.Mesh['bids'] = BIds
		
		
		self.Mesh['n_domain'] = len(DNames)
		self.Mesh['domain_names'] = DNames
		self.Mesh['domain_tags'] = DTags
		self.Mesh['dids'] = DIds

	def getTypeIndices(self):
		cnt = 0;
		bndry_idx = {};
		domain_idx = {};
		BNames = self.Mesh['bndry_names']
		DNames = self.Mesh['domain_names']
		for tag in (self.Mesh["bndry_tags"]):
			#print(tag)
			tmp = self.Mesh['tritype']
			tmpidx = (np.isin(tmp, tag ))
			tmpidx = np.where(tmpidx == True)
			#print(tmpidx[0])
			bndry_idx [BNames[cnt]] = tmpidx[0];
			cnt = cnt +1;
		self.Mesh["bndry_indx"] = bndry_idx

		cnt = 0;
		

		for tag in (self.Mesh["domain_tags"]):
			#print(tag)
			tmp = self.Mesh['tettype']
			#print(tmp,tag)
			#print('Vol Tags: ',np.unique(tmp))
			tmpidx = (np.isin(tmp, tag ))
			tmpidx = np.where(tmpidx == True)
			#print(tmpidx[0])
			domain_idx [DNames[cnt]] = tmpidx[0];
			cnt = cnt +1;
		#print(domain_idx)
		self.Mesh["domain_indx"] = domain_idx
		tetoftri = np.zeros((self.Mesh["ntri"],2),dtype=np.int);
		tetri = {}
		for nm in self.Mesh["bndry_names"]:
			faces = self.Mesh["bndry_indx"][nm]
			x = np.zeros(size(faces)[0],dtype=np.int)
			for i in range(size(faces)[0]):
				x[i]=(np.where(np.sum(np.isin(self.Mesh["tet2nodes"],np.unique(self.Mesh["tri2nodes"][faces[i],:])),1)==3)[0])[0]
			tetri[nm] = x
		for i in range(self.Mesh["ntri"]):
			x=(np.where(np.sum(np.isin(self.Mesh["tet2nodes"],np.unique(self.Mesh["tri2nodes"][i,:])),1)==3)[0])
			#print(size(x)[0])
			tetoftri[i,0:size(x)[0]]=x
		self.Mesh["tetoftri"] = tetri;
		#print(tetri)
	def generateEdge(self):
		elem = self.Mesh["tet2nodes"];
		node = self.Mesh["nodes"];
		tri = self.Mesh["tri2nodes"];
		NT = self.Mesh["ntet"];

		edgeseq = self.edgeseq
		edges = [];
		
		edges = np.row_stack((elem[:,edgeseq[0,:]],
						  elem[:,edgeseq[1,:]],
						  elem[:,edgeseq[2,:]],
						  elem[:,edgeseq[3,:]],
						  elem[:,edgeseq[4,:]],
						  elem[:,edgeseq[5,:]],))
		#print(elem[0,:])
		edges_dir_n = np.ones((NT*6,1))
		edges_dir_p = np.zeros((NT*6,1))
		#print(edges[0:10,:])
		#print(min(np.where(edges[:,0]>edges[:,1])[0]))
		edges_dir_n[np.where(edges[:,0]>edges[:,1])[0]] = 0
		edges_dir_p[np.where(edges[:,0]>edges[:,1])[0]] = 1
		
		edges = sort(edges,axis=1)
		
		edges_unique,indices = unique(edges,axis=0,return_inverse=True)

		#print(size(edges),size(edges_unique))
		

		elem2edge = np.reshape(indices,(NT,6),order ='F')
		edges_dir_p =  np.reshape(edges_dir_p,(NT,6),order ='F')
		edges_dir_n =  np.reshape(edges_dir_n,(NT,6),order ='F')
		#print(elem2edge_s[0:2,:])
		self.Mesh["edge2nodes"] = edges_unique;
		self.Mesh["tet2edges"] = elem2edge;
		self.Mesh["edges_dir_n"] = edges_dir_n;
		self.Mesh["edges_dir_p"] = edges_dir_p;


		triedges = np.row_stack((tri[:,[0,1]],
						  tri[:,[0,2]],
						  tri[:,[2,1]]))

		triedges = sort(triedges,axis=1)
		tri_ed_map = np.zeros(size(triedges)[0],dtype=np.int)
		for i in range(size(triedges)[0]):
			x = (edges_unique == triedges[i,:]).all(axis=1).nonzero()[0]
			tri_ed_map[i] = x

		tri2edges = np.reshape(tri_ed_map,(self.Mesh["ntri"],3),order ='F')
		self.Mesh["tri2edges"] = tri2edges;
		self.Mesh["nedges"] = size(edges_unique)[0]
		
	def generateFaces(self):
		elem = self.Mesh["tet2nodes"];
		node = self.Mesh["nodes"];
		tri = sort(self.Mesh["tri2nodes"],axis=1);
		NT = self.Mesh["ntet"];
		faceseq = self.faceseq
		faces = np.row_stack((elem[:,faceseq[0,:]],
							elem[:,faceseq[1,:]],
							elem[:,faceseq[2,:]],
							elem[:,faceseq[3,:]],))
		faces = sort(faces,axis=1)
		
		faces_unique,indices = unique(faces,axis=0,return_inverse=True)


		#print(size(edges),size(edges_unique))
		

		elem2face = np.reshape(indices,(NT,4),order ='F')
		self.Mesh["tet2faces"] = elem2face;

		tri_map = np.zeros(size(tri)[0],dtype=np.int)
		for i in range(size(tri)[0]):
			x = (faces_unique == tri[i,:]).all(axis=1).nonzero()[0]
			tri_map[i] = x

		#tri2faces = np.reshape(tri_map,(self.Mesh["ntri"],1),order ='F')
		self.Mesh["tri2faces"] = tri_map;
		self.Mesh["nfaces"] = size(faces_unique)[0]
		


	def sort_for_Hcurl(self):
		elem = self.Mesh["tet2nodes"]
		node = self.Mesh["nodes"]
		NT = self.Mesh["ntet"];
		elem = sort(elem,axis = 1);
		p1 = node[elem[:,0],:]
		p2 = node[elem[:,1],:]
		p3 = node[elem[:,2],:]
		p4 = node[elem[:,3],:]
		
		a = np.sum(np.multiply(np.cross(p2-p1,p3-p1),p4-p1),axis=1)
		idx = np.where(a<0)[0];
		orient = np.ones((NT,1));
		orient[idx] = -1;
		tmp = elem[idx,1]
		elem[idx,1] = elem[idx,2]
		elem[idx,2] = tmp
		self.Mesh["tet2nodes"] = elem;
		self.Mesh["tetorient"] = orient

	def generateJacobian(self):
		elem = self.Mesh["tet2nodes"]
		node = self.Mesh["nodes"]
		NT = self.Mesh["ntet"];

		p1 = node[elem[:,0],:]
		p2 = node[elem[:,1],:]
		p3 = node[elem[:,2],:]
		p4 = node[elem[:,3],:]
		

		x=np.column_stack((p1[:,0], p2[:,0], p3[:,0], p4[:,0]));
		y=np.column_stack((p1[:,1], p2[:,1], p3[:,1], p4[:,1]));
		z=np.column_stack((p1[:,2], p2[:,2], p3[:,2], p4[:,2]));
		#print((node[elem[100,0],:]))

		A = np.column_stack((x[:,0],y[:,0],z[:,0]))
		B = np.column_stack((x[:,1],y[:,1],z[:,1]))
		C = np.column_stack((x[:,2],y[:,2],z[:,2]))
		D = np.column_stack((x[:,3],y[:,3],z[:,3]))

		AB = B - A
		AC = C - A
		AD = D - A
		J = np.zeros((3,3,NT))
		
		J[:,0,:] = np.transpose(AB)
		J[:,1,:] = np.transpose(AC)
		J[:,2,:] = np.transpose(AD)
		J_Det = np.sum(np.multiply(AD,np.cross(AB,AC)),axis=1);
		J_I = MultiSolver(J,np.identity(3))
		J_IT = np.transpose(J_I,(1,0,2))
		self.Mesh["J"] = J
		self.Mesh["J_I"] = J_I
		self.Mesh["J_Det"] = J_Det
		self.Mesh["J_IT"] = J_IT
	
	def processMesh(self,verbose = True):
		print('Processing Mesh')
		#self.getTypeIndices();
		printprogress(3,7)


		self.sort_for_Hcurl()
		printprogress(4,7)

		self.generateEdge()
		printprogress(5,7)
		self.generateFaces()
		printprogress(6,7)
		
		self.generateJacobian()
		printprogress(7,7)

		if verbose == 0:
			return

		print('  Boundary Information\n------------------------------------')
		cnt =0;
		for nm in (self.Mesh["bndry_names"]):
			print("%15s :  %5d Triangles" %(nm,self.Mesh["bndry_indx"][nm].size))
			cnt=cnt+1

		cnt =0;

		print('\n	Domain Information\n----------------------------------')
		for nm in (self.Mesh["domain_names"]):
			print("%15s : %5d Tetrahedrons" %(nm,self.Mesh["domain_indx"][nm].size))
			cnt=cnt+1
		print("\n\n")
	def loadComsolMesh(self,fname,report = "*#?/",verbose = 0):
		print('Loading Mesh')
		self.readComsolFile(fname)
		printprogress(1,7)
		if(report == "*#?/"):
			self.readComsolReport(fname)
		else:
			self.readComsolReport(report)
		printprogress(2,7)
		self.Mesh["filename"] = fname;
		self.getTypeIndices();
		printprogress(3,7)


		self.sort_for_Hcurl()
		printprogress(4,7)

		self.generateEdge()
		printprogress(5,7)
		self.generateFaces()
		printprogress(6,7)
		
		self.generateJacobian()
		printprogress(7,7)

		if verbose == 0:
			return

		print('  Boundary Information\n------------------------------------')
		cnt =0;
		for nm in (self.Mesh["bndry_names"]):
			print("%15s :  %5d Triangles" %(nm,self.Mesh["bndry_indx"][nm].size))
			cnt=cnt+1

		cnt =0;

		print('\n	Domain Information\n----------------------------------')
		for nm in (self.Mesh["domain_names"]):
			print("%15s : %5d Tetrahedrons" %(nm,self.Mesh["domain_indx"][nm].size))
			cnt=cnt+1
		print("\n\n")


def MultiSolver(M, RHS):
		[m,n,p]=M.shape
		RHS=np.tile(RHS,p);
		RHS=np.transpose(RHS)
		
		q=RHS.shape[1];
		
		Q=np.array(range(0,m*p))
		Q=np.reshape(Q,[p,1,m])
		Q=np.transpose(Q,[2,1,0]);
		R=np.array(range(0,n*p))
		R=np.reshape(R,[p,1,n])
		R=np.transpose(R,[1,2,0]);

		I = np.tile(Q,[1,n,1]);
		J = np.tile(R,[m,1,1]); 
		
		A=sps.csr_matrix((M.flatten('F'),(I.flatten('F'),J.flatten('F')))	,dtype=np.float)
			
		
		
		X=spsl.spsolve(A,RHS[:,0]);
		Y=spsl.spsolve(A,RHS[:,1]);
		Z=spsl.spsolve(A,RHS[:,2]);

		X=np.column_stack([X,Y,Z]);


		
		X=np.reshape(X, [p,9]);
		X=np.reshape(X, [p,3,3]);
		X=np.transpose( X, (1,2,0) )

		return X

