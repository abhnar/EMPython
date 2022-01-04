from numpy import unique,max,min,abs,zeros,intersect1d,hstack,transpose,ones,array,\
					float,complex,pi,delete,where,imag,real,sort,complex128,mean, sqrt

from numpy.linalg import norm,det,lstsq,solve,eig
import traceback
from Utils import triquad
from time import time
from types import SimpleNamespace
def computePortMatrices(nds,tri2nd,tri2ed,pectri2nd,pectri2ed,epsilonrs):
	#Works only on ports in Z direction
	
	
	MATRICES =  SimpleNamespace(Stt = None, Ttt = None, Tzt = None, Ttz = None, Tzz = None, tri2edges = None, pec_edge_index = None, automatic = True)
	
	try:
		edgeseq = array(((0, 1), (0, 2), (2, 1)))

		edge_index = unique(tri2ed.flatten())
		nodes_index = unique(tri2nd.flatten())
		pec_edge_index = unique(pectri2ed.flatten())
		pec_node_index = unique(pectri2nd.flatten())
		pec_edge_index = intersect1d(pec_edge_index,edge_index)
		pec_node_index = intersect1d(pec_node_index,nodes_index)

		ed_hashtable = {}
		cnt = 0
		
		for ed in edge_index:
			
			ed_hashtable[ed] = cnt
			cnt = cnt+1

		ntri = tri2ed.shape[0]
		tri2edge = zeros(tri2ed.shape)
		for i in range(ntri):
			for j in range(3):
				tri2edge[i,j] = ed_hashtable[tri2ed[i,j]]

		
		nd_hashtable = {}

		cnt = 0
		NE = edge_index.size
		NN = nodes_index.size
		#global2local_nodes = [0]*NN
		nodes = zeros((NN,3))
		for nd in nodes_index:
			
			nd_hashtable[nd] = cnt
			#global2local_nodes[cnt] = nd
			nodes[cnt,:] = nds[nd,:]
			cnt = cnt + 1

		for i in range(pec_node_index.size):
			pec_node_index[i] = nd_hashtable[pec_node_index[i]]
		for i in range(pec_edge_index.size):
			pec_edge_index[i] = ed_hashtable[pec_edge_index[i]]


		tri2node = zeros(tri2ed.shape)
		for i in range(ntri):
			for j in range(3):
				tri2node[i,j] = nd_hashtable[tri2nd[i,j]]
		tri2node = tri2node.astype(int)
		pec_edge_index = pec_edge_index.astype(int)
		pec_node_index = pec_node_index.astype(int)
		tri2edge = tri2edge.astype(int)

		tmp = tri2node[:, [1, 2, 1]] - tri2node[:, [0, 0, 2]];
		sign_edge = tmp / abs(tmp);
		sign_edge = transpose(sign_edge)
		
		Stt = zeros((NE, NE), dtype=float)
		Ttt = zeros((NE, NE), dtype=float)
		Tzt = zeros((NN, NE), dtype=float)
		Tzz = zeros((NN, NN), dtype=float)
		Tzz_i = zeros((NN, NN), dtype=float);
		Tzz_f = zeros((NN, NN), dtype=float);
		Stt_i = zeros((NE, NE), dtype=float);
		Stt_f = zeros((NE, NE), dtype=float);

		Am = zeros((ntri,3))
		Bm = zeros((ntri,3))
		Cm = zeros((ntri,3))
		Dm = zeros((ntri,3))
		Lm = zeros((ntri,3))
		A = zeros((ntri,1))
		x = zeros((3,1))
		y = zeros((3,1))
		ONES = ones((3, 1))
		for e in range(ntri):
			x[:,0] = nodes[tri2node[e,:],0]
			y[:,0] = nodes[tri2node[e,:],1]
			for m in range(3):
				i = edgeseq[m, 0]
				j = edgeseq[m, 1]
				
				Lm[e,m] = norm([(x[i] - x[j], y[i] - y[j])])


				a = [x[1] * y[2] - y[1] * x[2], x[2] * y[0] - y[2] * x[0], x[0] * y[1] - y[0] * x[1]]
				b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
				c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]
				
				I = (hstack((ONES, x, y)))
				A[e] = abs(0.5 * det(I))

				Am[e,m] = a[i] * b[j] - a[j] * b[i];
				Bm[e,m] = c[i] * b[j] - c[j] * b[i];
				Cm[e,m] = a[i] * c[j] - a[j] * c[i];
				Dm[e,m] = -Bm[e,m]

		

		funx = lambda x, y: x
		funx2 = lambda x, y: x ** 2
		funy = lambda x, y: y
		funy2 = lambda x, y: y ** 2;
		funxy = lambda x, y: x * y;
		t1 = time()
		for e in range(ntri):
			x = nodes[tri2node[e,:],0]
			y = nodes[tri2node[e,:],1]

			a = [x[1] * y[2] - y[1] * x[2], x[2] * y[0] - y[2] * x[0], x[0] * y[1] - y[0] * x[1]]
			b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
			c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]

			integx = triquad(funx, x[0], x[1], x[2], y[0], y[1], y[2]);
			integy = triquad(funy, x[0], x[1], x[2], y[0], y[1], y[2]);
			integx2 = triquad(funx2, x[0], x[1], x[2], y[0], y[1], y[2]);
			integy2 = triquad(funy2, x[0], x[1], x[2], y[0], y[1], y[2]);
			integxy = triquad(funxy, x[0], x[1], x[2], y[0], y[1], y[2]);
			epsr = epsilonrs[e]


			

			for m in range(3):
				for n in range(3):
					i=m
					j=n
					I2 = sign_edge[m, e] * sign_edge[n, e] * (
							(Bm[e,m] * Bm[e,n] * Lm[e,m] * Lm[e,n]) / (16 * A[e] ** 4) - (Bm[e,m] * Dm[e,n] * Lm[e,m] * Lm[e,n]) / (16 * A[e] ** 4) - (
							Bm[e,n] * Dm[e,m] * Lm[e,m] * Lm[e,n]) / (16 * A[e] ** 4) + (Dm[e,m] * Dm[e,n] * Lm[e,m] * Lm[e,n]) / (16 * A[e] ** 4)) * A[e];
					I3 = sign_edge[m, e] * sign_edge[n, e] * Lm[e,m] * Lm[e,n] * (
							(Am[e,m] * Am[e,n]) * A[e] + (Cm[e,m] * Cm[e,n]) * A[e] + (Bm[e,m] * Bm[e,n] * integy2) + (Dm[e,m] * Dm[e,n] * integx2) + (
							Am[e,m] * Bm[e,n] * integy) + (Am[e,n] * Bm[e,m] * integy) + (Cm[e,m] * Dm[e,n] * integx) + (
									Cm[e,n] * Dm[e,m] * integx)) / (16 * A[e] ** 4);
					I4 = (Am[e,n] * b[i] * sign_edge[n, e] * Lm[e,n]) / (8 * A[e] ** 3) * A[e] + (Cm[e,n] * c[i] * sign_edge[n, e] * Lm[e,n]) / (
							8 * A[e] ** 3) * A[e] + (Bm[e,n] * b[i] * sign_edge[n, e] * Lm[e,n] * integy) / (8 * A[e] ** 3) + (
								 Dm[e,n] * c[i] * sign_edge[n, e] * Lm[e,n] * integx) / (8 * A[e] ** 3);
					I5 = (a[i] * a[j]) / (4 * A[e] ** 2) * A[e] + (a[i] * b[j]) / (4 * A[e] ** 2) * integx + (
							a[j] * b[i] * integx) / (4 * A[e] ** 2) + (a[i] * c[j] * integy) / (4 * A[e] ** 2) + (
								 a[j] * c[i] * integy) / (4 * A[e] ** 2) + (b[i] * b[j] * integx2) / (4 * A[e] ** 2) + (
								 c[i] * c[j] * integy2) / (4 * A[e] ** 2) + (b[i] * c[j] * integxy) / (4 * A[e] ** 2) + (
								 b[j] * c[i] * integxy) / (4 * A[e] ** 2);
					
					#Stt[tri2edge[e, m], tri2edge[e, n]] = Stt[tri2edge[e, m], tri2edge[e, n]] + (
					#		I2 - k0 ** 2 * epsr * I3);
					Stt_i[tri2edge[e, m], tri2edge[e, n]] = Stt_i[tri2edge[e, m], tri2edge[e, n]] + I2 
					Stt_f[tri2edge[e, m], tri2edge[e, n]] = Stt_f[tri2edge[e, m], tri2edge[e, n]] -epsr * I3

					Ttt[tri2edge[e, m], tri2edge[e, n]] = Ttt[tri2edge[e, m], tri2edge[e, n]] + I3
					Tzt[tri2node[e, i], tri2edge[e, n]] = Tzt[tri2node[e, i], tri2edge[e, n]] + I4;

					#Tzz[tri2node[e, i], tri2node[e, j]] = Tzz[tri2node[e, i], tri2node[e, j]] + \
					#		(b[i] * b[j] + c[i] * c[j]) / (4 * A[e] ** 2) * A[e] - k0 ** 2 * epsr * I5

					Tzz_i[tri2node[e, i], tri2node[e, j]] = Tzz_i[tri2node[e, i], tri2node[e, j]] +  (b[i] * b[j] + c[i] * c[j]) / (4 * A[e] ** 2) * A[e] 
					Tzz_f[tri2node[e, i], tri2node[e, j]] = Tzz_f[tri2node[e, i], tri2node[e, j]]  -  epsr * I5
		
		

		

		Stt_i = delete(Stt_i, pec_edge_index, axis=0)
		Stt_i = delete(Stt_i, pec_edge_index, axis=1)
		Stt_f = delete(Stt_f, pec_edge_index, axis=0)
		Stt_f = delete(Stt_f, pec_edge_index, axis=1)

		Ttt = delete(Ttt, pec_edge_index, axis=0)
		Ttt = delete(Ttt, pec_edge_index, axis=1)
		
		

		Tzz_i = delete(Tzz_i, pec_node_index, axis=0)
		Tzz_i = delete(Tzz_i, pec_node_index, axis=1)
		Tzz_f = delete(Tzz_f, pec_node_index, axis=0)
		Tzz_f = delete(Tzz_f, pec_node_index, axis=1)

		Tzt = delete(Tzt, pec_node_index, axis=0)
		Tzt = delete(Tzt, pec_edge_index, axis=1)
		Ttz = transpose(Tzt)

		MATRICES.Stt_i = Stt_i
		MATRICES.Stt_f = Stt_f
		MATRICES.Tzz_i = Tzz_i
		MATRICES.Tzz_f = Tzz_f
		MATRICES.Ttt = Ttt
		MATRICES.Tzt = Tzt
		MATRICES.Ttz = Ttz
		MATRICES.tri2edge = tri2edge
		MATRICES.tri2node = tri2node
		MATRICES.pec_edge_index = pec_edge_index
		MATRICES.nodes = nodes
		MATRICES.NE = NE
		
		return MATRICES
	except:
		print("Error in Portmode Analysis")
		
		traceback.print_exc()

def portModeAnalysis(MATRICES,f):		
	mu0=4*pi*1e-7;
	eps0=8.85418781762039e-12;
	edgeseq = array(((0, 1), (0, 2), (2, 1)))
	TRIQUAD = SimpleNamespace(weights = [0.225, 0.13239415278851, 0.13239415278851, 0.13239415278851, 0.12593918054483, 0.12593918054483, 0.12593918054483],
							 xi = [0.3333,	0.4701,	0.4701,	0.0597,	0.1013,	0.1013,	0.7974],
							 eta = [0.3333,	0.4701,	0.0597,	0.4701,	0.1013,	0.7974,	0.1013], Nq = 7, automatic=True)
	
	try:	
		f = f * 1e9;
		c = 1/sqrt(mu0 * eps0);
		wl = c / f;
		k0 = 2 * pi / wl;

		Stt = MATRICES.Stt_i + k0**2*MATRICES.Stt_f
		Tzz = MATRICES.Tzz_i + k0**2*MATRICES.Tzz_f

		
		#status.emit("K0 :" + str(k0 ))
		

		t1 = time()
		temp = lstsq(Tzz, MATRICES.Tzt, rcond=None)
		Btt = MATRICES.Ttz.dot(temp[0]) - MATRICES.Ttt
		C = solve(Btt, Stt)
		res = eig(C)
		#status.emit("Eigen SOlve TIme :" + str(time() - t1 ))
		eigvals = res[0]
		evec = res[1]
		

		beta1 = sqrt(eigvals)

		ix = where(abs(imag(beta1)) < 0.0001)[0];
		beta = beta1[ix]
		beta = beta[abs((real(beta))) > 0]
		beta[::-1].sort()
		##print(beta)
		NE = MATRICES.NE
		pec_edge_index = MATRICES.pec_edge_index
		#print("no of modes: ", beta.shape)
		if (beta.size != 0):
			Kz10 = (max(beta))
			#Kz10 = beta[-4]
			#print(beta[-1],Kz10,k0)
			v = evec[:, beta1 == Kz10]
			ins = array(range(NE))
			ins[pec_edge_index] = -1
			ins = where(ins != -1)[0]
			v1 = zeros((NE, 1), dtype=complex128)
			v1[ins] = v
			v = v1 

			

		#status.emit(str(max(Tzz)) +"," + str(max(Tzt))+"," + str(max(Ttz))+"," + str(max(Ttt))+"," + str(max(Stt))+"," + str(max(abs(v))))
		

		tri2edge = MATRICES.tri2edge
		tri2node = MATRICES.tri2node

		tmp = tri2node[:, [1, 2, 1]] - tri2node[:, [0, 0, 2]];
		sign_edge = tmp / abs(tmp);
		sign_edge = transpose(sign_edge)
		#MATRICES.pec_edge_index = pec_edge_index	
		nodes = MATRICES.nodes
		x = zeros((3,1))
		y = zeros((3,1))
		ONES = ones((3, 1))

		weights = TRIQUAD.weights
		xi = TRIQUAD.xi
		eta = TRIQUAD.eta
		Nq = TRIQUAD.Nq
		Y = zeros((Nq,3),dtype=float)
		Phi = [None]*tri2edge.shape[0]
		PLOTDATA = zeros((tri2edge.shape[0],4),dtype=complex)

		Modes = [None]*beta.shape[0]
		Kz10s = [0]*beta.shape[0]
		if (beta.size != 0):
			for k in range(beta.shape[0]):
				Kz10s[k] = beta[k]
				
				#print(beta[-1],Kz10,k0)
				v2 = evec[:, beta1 == Kz10s[k]]
				ins = array(range(NE))
				ins[pec_edge_index] = -1
				ins = where(ins != -1)[0]
				v1 = zeros((NE, 1), dtype=complex128)
				v1[ins] = v2
				v2 = v1 

				
				Modes[k] = v2
		Phis = [None]*beta.shape[0]
		
		
		for e in range(tri2edge.shape[0]):
			x[:,0] = nodes[tri2node[e,:],0]
			y[:,0] = nodes[tri2node[e,:],1]

			xyz = nodes[tri2node[e,:],:]
			for w in range(Nq):
				Y[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]
			

			a = [x[1] * y[2] - y[1] * x[2], x[2] * y[0] - y[2] * x[0], x[0] * y[1] - y[0] * x[1]]
			b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
			c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]
			
			I = (hstack((ONES, x, y)))
			A = abs(0.5 * det(I))
			L = [None] * 3
			DL = [None] * 3

			W = [None] * 3
			for i in range(3):
				L[i] = lambda x, y, i=i: 1 / (2 * A) * (a[i] + x * b[i] + y * c[i])
				DL[i] = lambda x, y, i=i: 1 / (2 * A) * array([b[i], c[i]])

			for i in range(3):
				i1 = edgeseq[i, 0]
				i2 = edgeseq[i, 1]
				lm = norm(array((x[i1] - x[i2], y[i1] - y[i2])))

				W[i] = lambda x, y, i=i, i1=i1, i2=i2, lm=lm: sign_edge[i,e] * lm * (
						L[i1](x, y) * DL[i2](x, y) - L[i2](x, y) * DL[i1](x, y));

			Ef_vec = lambda x,y:v[tri2edge[e,0]] * W[0](x, y) + v[tri2edge[e,1]]*  W[1](x, y) + v[tri2edge[e,2]] * W[2](x, y)
			#print(Y.shape)
			Phi[e] = Ef_vec(Y[:,0],Y[:,1])
			#print(DL[0](1, 1))
			xm = mean(xyz[:,0])
			ym = mean(xyz[:,1])
			
			val = Ef_vec(xm,ym)
			PLOTDATA[e,:]=[xm,ym,val[0],val[1]]
		port_modes = [None]*beta.shape[0]
		for k in range(beta.shape[0]):
			PLTD = zeros((tri2edge.shape[0],4),dtype=complex)
			Phis_tmp = [None]*tri2edge.shape[0]
			v= Modes[k]

			for e in range(tri2edge.shape[0]):
				x[:,0] = nodes[tri2node[e,:],0]
				y[:,0] = nodes[tri2node[e,:],1]

				xyz = nodes[tri2node[e,:],:]
				for w in range(Nq):
					Y[w,:] = xyz[0,:] + (xyz[1,:]-xyz[0,:])*xi[w] + (xyz[2,:]-xyz[0,:])*eta[w]
				

				a = [x[1] * y[2] - y[1] * x[2], x[2] * y[0] - y[2] * x[0], x[0] * y[1] - y[0] * x[1]]
				b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
				c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]
				
				I = (hstack((ONES, x, y)))
				A = abs(0.5 * det(I))
				L = [None] * 3
				DL = [None] * 3

				W = [None] * 3
				for i in range(3):
					L[i] = lambda x, y, i=i: 1 / (2 * A) * (a[i] + x * b[i] + y * c[i])
					DL[i] = lambda x, y, i=i: 1 / (2 * A) * array([b[i], c[i]])

				for i in range(3):
					i1 = edgeseq[i, 0]
					i2 = edgeseq[i, 1]
					lm = norm(array((x[i1] - x[i2], y[i1] - y[i2])))

					W[i] = lambda x, y, i=i, i1=i1, i2=i2, lm=lm: sign_edge[i,e] * lm * (
							L[i1](x, y) * DL[i2](x, y) - L[i2](x, y) * DL[i1](x, y));

				Ef_vec = lambda x,y:v[tri2edge[e,0]] * W[0](x, y) + v[tri2edge[e,1]]*  W[1](x, y) + v[tri2edge[e,2]] * W[2](x, y)
				#print(Y.shape)
				Phis_tmp[e] = Ef_vec(Y[:,0],Y[:,1])
				#print(DL[0](1, 1))
				xm = mean(xyz[:,0])
				ym = mean(xyz[:,1])
				
				val = Ef_vec(xm,ym)
				PLTD[e,:]=[xm,ym,val[0],val[1]]
			Phis[k] = Phis_tmp
			port_modes[k] = PLTD

		#Phis[0] = Phi
		#from scipy.io import loadmat as load, savemat as save
		#save('pltdata.mat',{'pltdata':PLOTDATA,'pormodes':port_modes})
		PortField =  SimpleNamespace(automatic=True)
		PortField.Modes = Phis
		PortField.Kz10s = Kz10s
		PortField.nmodes = beta.shape[0]
		#PortField.nmodes = 1

				
		
		return (Kz10,Phi,PortField)	



			



		
			 


		
	except Exception as e:
		print("Error in Portmode Analysis")

		
		traceback.print_exc()
		
	

