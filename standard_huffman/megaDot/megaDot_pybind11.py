import time
import numpy as np
from termcolor import colored
import libexample as megaDot

n = 100000		# Provato numero dispari per check sulla parte rimanente del vettore
m = 100000
identmatrix = np.identity(n)
constvect = np.ones(n) * 2
res_array_ref = np.zeros(n)
res_array1 = np.zeros(n)
res_array2 = np.zeros(n)
res_array3 = np.zeros(n)
res_array4 = np.zeros(n)
res_array5 = np.zeros(n)
res_array6 = np.zeros(n)
res_array7 = np.zeros(n)

print("Prova import tramite pybind11 - n:", n)

# Reference: numpy dot product
time1 = time.time()
res_array_ref = np.dot(constvect, identmatrix)
py_time = time.time() - time1
print("py running time in seconds: {:.6f}".format(py_time))

# c: original implementation
time1 = time.time()
status = megaDot.cpp_matDot_original(identmatrix, constvect, n, m, res_array2)
timeC = time.time() - time1
print("cpp dot original - running time in seconds: {:.6f}".format(timeC) + colored(" ({:.4f}x)".format(py_time / timeC), "green"))

# c: openmp
time1 = time.time()
status = megaDot.cpp_matDot_original_openmp(identmatrix, constvect, n, m, res_array3)
timeComp = time.time() - time1
print("cpp openmp - running time in seconds: {:.6f}".format(timeComp) + colored(" ({:.4f}x)".format(py_time / timeComp), "green"))

# c: autovect via pragma simd
time1 = time.time()
status = megaDot.cpp_matDot_pragmasimd(identmatrix, constvect, n, m, res_array4)
timeCauto = time.time() - time1
print("cpp autovect - running time in seconds: {:.6f}".format(timeCauto) + colored(" ({:.4f}x)".format(py_time / timeCauto), "green"))

# c: loop unrolling
time1 = time.time()
status = megaDot.cpp_matDot_loopunroll(identmatrix, constvect, n, m, res_array5)
timeCunr = time.time() - time1
print("cpp loop unrolling - running time in seconds: {:.6f}".format(timeCunr) + colored(" ({:.4f}x)".format(py_time / timeCunr), "green"))

# c: avx2
time1 = time.time()
status = megaDot.cpp_matDot_avx2(identmatrix, constvect, n, m, res_array6)
timeCavx2 = time.time() - time1
print("cpp avx2 - running time in seconds: {:.6f}".format(timeCavx2) + colored(" ({:.4f}x)".format(py_time / timeCavx2), "green"))

# c: avx2 + openmp
time1 = time.time()
status = megaDot.cpp_matDot_avx2_openmp4(identmatrix, constvect, n, m, res_array7)
timeCavx2omp = time.time() - time1
print("cpp avx2 + openmp - running time in seconds: {:.6f}".format(timeCavx2omp) + colored(" ({:.4f}x)".format(py_time / timeCavx2omp), "green"))


#print(np.array_equal(res_array_ref, res_array1))
print(np.array_equal(res_array_ref, res_array2))
print(np.array_equal(res_array_ref, res_array3))
print(np.array_equal(res_array_ref, res_array4))
print(np.array_equal(res_array_ref, res_array5))
print(np.array_equal(res_array_ref, res_array6))
print(np.array_equal(res_array_ref, res_array7))
