from types import SimpleNamespace
import numpy as np
from ctypes import cdll, c_int, POINTER, c_double, c_int64, c_void_p
from scipy.sparse import triu
import sys

class MKL_sparse():
	errdata = dict({(-1,'input inconsistent'),(-2,'not enough memory'),(-3,'reordering problem'),(-4,'Zero pivot, numerical factorization or iterative refinement problem. If the error appears during the solution phase, try to change the pivoting perturbation (iparm[9]) and also increase the number of iterative refinement steps. If it does not help, consider changing the scaling, matching and pivoting options (iparm[10], iparm[12], iparm[20])'),\
					(-5,'unclassified (internal) error'),(-6,'reordering failed (matrix types 11 and 13 only)'),\
					(-7,'diagonal matrix is singular'),(-8,'32-bit integer overflow problem'),(-9,'not enough memory for OOC'),
					(-10,'error opening OOC files'),(-11,'read/write error with OOC files'),(-12,'(pardiso_64 only) pardiso_64 called from 32-bit library'),
					(-13,'interrupted by the (user-defined) mkl_progress function'),(-15,'internal error which can appear for iparm[23]=10 and iparm[12]=1. Try switch matching off (set iparm[12]=0 and rerun.)')})
	Phase = SimpleNamespace(Aalysis = 11, Num_Fact = 22, Solve = 33 )
	MatrixType = SimpleNamespace(real_struct_sym = c_int64(1), real_sym_pd = c_int64(2), real_sym_ind =  c_int64(-2), \
								 complex_struct_sym = c_int64(3), complex_her_pd = c_int64(4), \
								 complex_herm_ind = c_int64(-4), complex_sym = c_int64(6), real_unsym = c_int64(11), complex_unsym = c_int64(13))


	def __init__(self):
		try:
			self.dll = cdll.LoadLibrary("./mkl/mklsolve.dll")
			#self.dll = cdll.LoadLibrary("D:/del/mklsolve.dll")
			

		except OSError as e:
			print("Critical Error: Unable to load 'mklsolve.dll' file.")
			sys.exit(1)
	def solve_sym(self,A,b):
		A = triu(A,format='csr')
		if(A.dtype==np.complex128 ):
			return self.solve(A,b,self.MatrixType.complex_sym)
		else:
			return self.solve(A,b,self.MatrixType.real_sym_ind)

	def solve(self,A,b, mattype = c_int64(3)):
		n = A.shape[0]
		type = 'real'
		if(A.dtype==np.complex128 ):
			b = b.astype(np.complex128)
			type = 'complex'
		elif(b.dtype==np.complex128 ):
			A = A.astype(np.complex128)
			type = 'complex'

		data = A.data

		indices = A.indices
		indptr = A.indptr
		indices = indices+1
		indptr = indptr+1

		indptr = np.array(indptr, dtype = np.int64)
		indices = np.array(indices, dtype = np.int64)
		
		
		ja = indices.ctypes.data_as(POINTER(c_int64))
		ia = indptr.ctypes.data_as(POINTER(c_int64))
		

		if type == 'real':
			#print("Matrix is real")
			a = data.ctypes.data_as(POINTER(c_double))
			x = np.zeros(n,order='C',dtype=np.double)
			b = np.array(b, dtype = np.double)
			ptr_b = b.ctypes.data_as(POINTER(c_double))
			ptr_x = x.ctypes.data_as(POINTER(c_double))
			
			info = self.dll.real( a,ia, ja, ptr_b, ptr_x,  c_int64(n), mattype)
		elif type == 'complex':
			a=data.ctypes.data_as(POINTER(c_void_p))
			x=np.zeros(n,order='C',dtype=np.complex128)
			ptr_b=b.ctypes.data_as(POINTER(c_void_p))
			ptr_x=x.ctypes.data_as(POINTER(c_void_p))
			info=self.dll.complex( a,ia, ja, ptr_b, ptr_x,  c_int64(n),  mattype )
		if info != 0:
			x = None
		return x

