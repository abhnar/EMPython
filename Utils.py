from ctypes import c_int,c_double,cdll,POINTER,byref,c_void_p
import numpy as np
import scipy.sparse as sps
from scipy import stats



#mkl = cdll.LoadLibrary("D:\\ProgramData\Anaconda3\\Library\\bin\\mkl_rt.dll")
from inspect import getframeinfo, stack

def dprint(message):
    caller = getframeinfo(stack()[1][0])
    print("%s:%d - %s" % (caller.filename.split('\\')[-1], caller.lineno, str(message))) # python3 syntax print

def spsolveinterface(A,b,mattype):
	dll = cdll.LoadLibrary(".\\mklpythoninterface.dll")
	#A=triu(A,format='csr')
	n=A.shape[0]
	#print(A.dtype)
	#b=b.toarray();
	#b=np.ones((n,1),dtype=np.double)
   # print(b.dtype)
	ret=np.zeros((14,1),dtype=np.double)
	mattype = 13
	ptr_ret=ret.ctypes.data_as(POINTER(c_int))
	#print(A.dtype,b.dtype)
	if(A.dtype==np.complex128 and b.dtype==np.complex128):
		#print("complex")
		data=A.data
		indices=A.indices
		indptr=A.indptr
		indices=indices+1
		indptr=indptr+1
# =============================================================================
#		 np.savetxt('ja.txt',indices,fmt='%.2f')
#		 np.savetxt('ia.txt',indptr,fmt='%.2f')
#		 np.savetxt('a.txt',data,fmt='%.2f')
# =============================================================================
		
		a=data.ctypes.data_as(POINTER(c_void_p))
		ja=indices.ctypes.data_as(POINTER(c_int))
		ia=indptr.ctypes.data_as(POINTER(c_int))
		x=np.zeros((n,1),order='C',dtype=np.complex128)
		ptr_b=b.ctypes.data_as(POINTER(c_void_p))
		ptr_x=x.ctypes.data_as(POINTER(c_void_p))
		info=dll.spsolve_pardiso( a,ia, ja, ptr_b, ptr_x,  c_int(n), ptr_ret,c_int(mattype) )
	elif(A.dtype==np.float64 and b.dtype==np.float64):
		data=A.data
		indices=A.indices
		indptr=A.indptr
		indices=indices+1
		indptr=indptr+1
		a=data.ctypes.data_as(POINTER(c_double))
		ja=indices.ctypes.data_as(POINTER(c_int))
		ia=indptr.ctypes.data_as(POINTER(c_int))
		x=np.zeros((n,1),order='C',dtype=np.double)
		ptr_b=b.ctypes.data_as(POINTER(c_double))
		ptr_x=x.ctypes.data_as(POINTER(c_double))
		info=dll.spsolve_real( a,ia, ja, ptr_b, ptr_x,  c_int(n), ptr_ret,c_int(mattype) )
	else:
		print("Incompatible types")
		print(A.dtype,b.dtype)
		x = 0
		
	return x


def triquad(f, x1, x2, x3, y1, y2, y3):
	xw = np.array([[4.459484909159700e-01, 4.459484909159700e-01, 2.233815896780100e-01],
				   [4.459484909159700e-01, 1.081030181680700e-01, 2.233815896780100e-01],
				   [1.081030181680700e-01, 4.459484909159700e-01, 2.233815896780100e-01],
				   [9.157621350976999e-02, 9.157621350976999e-02, 1.099517436553200e-01],
				   [9.157621350976999e-02, 8.168475729804600e-01, 1.099517436553200e-01],
				   [8.168475729804600e-01, 9.157621350976999e-02, 1.099517436553200e-01]])
	A = abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0;

	z = 0.0

	for j in range(6):
		x = x1 * (1 - xw[j, 0] - xw[j, 1]) + x2 * xw[j, 0] + x3 * xw[j, 1]
		y = y1 * (1 - xw[j, 0] - xw[j, 1]) + y2 * xw[j, 0] + y3 * xw[j, 1];
		z = z + f(x, y) * xw[j, 2];

	z = A * z
	return z
def printprogress(i,max, value =0):
	k=int(i/max*10);
	str_h ="\u2588";
	str_l =" ";
	if not (i==max):
		print('\tProgress (%2.3f ): %3.0f'%(value,i/max*100),'% |'+str(str_h*k*2)+str((10-k)*2*str_l)+'|', end="\r")
	else:
		print('\tProgress		 :  100 %',str_h*10*2,'	')

def ismember(a, b):
	bind = {}
	for i, elt in enumerate(b):
		if elt not in bind:
			bind[elt] = True
	return [bind.get(itm, False) for itm in a] 
def delete_from_csr(mat, row_indices=[], col_indices=[]):
	"""
	Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) form the CSR sparse matrix ``mat``.
	WARNING: Indices of altered axes are reset in the returned matrix
	"""
	if not isinstance(mat, sps.csr_matrix):
		raise ValueError("works only for CSR format -- use .tocsr() first")

	rows = []
	cols = []

	#if row_indices:
	rows = list(row_indices)
	#if col_indices.any():
	cols = list(col_indices)

	if len(rows) > 0 and len(cols) > 0:
		row_mask = np.ones(mat.shape[0], dtype=bool)
		row_mask[rows] = False
		col_mask = np.ones(mat.shape[1], dtype=bool)
		col_mask[cols] = False
		return mat[row_mask][:,col_mask]
	elif len(rows) > 0:
		mask = np.ones(mat.shape[0], dtype=bool)
		mask[rows] = False
		return mat[mask]
	elif len(cols) > 0:
		mask = np.ones(mat.shape[1], dtype=bool)
		mask[cols] = False
		return mat[:,mask]
	else:
		return mat

def dot(A,B, ax = 1):
	tmp = np.sum(A*B, axis = ax);
	return 	tmp
