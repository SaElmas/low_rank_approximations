import scipy
from scipy import io
from scipy.sparse import csr_matrix
from numpy import linalg
import time



start = time.time()
A = scipy.io.mmread("data/bcspwr10.mtx")
Q, S, V = linalg.svd(A.todense())
end = time.time()

print(end-start)

