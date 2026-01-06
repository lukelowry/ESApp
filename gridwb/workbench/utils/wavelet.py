
from numpy import block, diag, real, imag
from scipy.linalg import schur

def takagi(M):
   n = M.shape[0]
   D, P = schur(block([[-real(M),imag(M)],[imag(M),real(M)]]))
   pos = diag(D) > 0
   Sigma = diag(D[pos,pos])
   # Note: The arithmetic below is technically not necessary
   U = P[n:,pos] + 1j*P[:n,pos]
   return U, Sigma.diagonal()

