#include <iostream>
#include <string>
#include <thread>
#include <chrono>
#include <immintrin.h>
#include <cstring>
#include <omp.h>
#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/numpy.h"

int main(void) {
	std::cout << "LaLaLa" << std::endl;
	return 0;
}

void printAMessage() {
	std::cout << "stampo da C++" << std::endl;
	pybind11::object os = pybind11::module_::import("os");
	std::cout << "Ho importato il package os di Python da dentro C++" << std::endl;
	std::cout << os.attr("sep").cast<std::string>() << std::endl;
}

int cpp_matDot_avx2(pybind11::array_t<double> npMat, pybind11::array_t<double> npVec, int n, int m, 
        pybind11::array_t<double> & npRes) {
    
    pybind11::buffer_info matrBufInfo = npMat.request();
    pybind11::buffer_info vectBufInfo = npVec.request();
    pybind11::buffer_info reslBufInfo = npRes.request();

    double *matrix  = static_cast<double *>(matrBufInfo.ptr);
    double *vector  = static_cast<double *>(vectBufInfo.ptr);
    double *results = static_cast<double *>(reslBufInfo.ptr);

	// Non è possibile dire in anticipo se il puntatore alla matrice sia allineato al margine richiesto
	// dalle istruzioni avx2. Idem per puntatore al vettore,
	// quindi devo usare _mm256_loadu_pd invece che _mm256_load_pd (random seg fault altrimenti)
	// Peccato, perché sarebbe ancora più veloce!
	for (size_t i = 0; i < m; i++) {
		// Dot product manuale, 4 double alla volta
		// Questo perchè per i double non esiste l'istruzione equivalente a _mm256_dp_ps :(
		__m256d sum_vec = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
		for(size_t j = 0; j < (n/4)*4; j+=4) {
			__m256d x = _mm256_loadu_pd(matrix + i*n + j);
			__m256d y = _mm256_loadu_pd(vector + j);
			__m256d z = _mm256_mul_pd(x,y);
			sum_vec = _mm256_add_pd(sum_vec, z);
		}
		// resto + accumulo dentro result[i]
		results[i] = 0.0;
		for(int j = n-n%4; j < n; ++j)
			results[i] += matrix[i*n + j] * vector[j];
		// La reduction finale si potrebbe fare con quei tre intrinsec,
		// ma non si guadagna niente. Anzi, forse il compilatore ottimizza di suo
		results[i] += (sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3]);
	}
	return 0;
}

int cpp_matDot_avx2_openmp4(pybind11::array_t<double> npMat, pybind11::array_t<double> npVec, int n, int m, 
        pybind11::array_t<double> & npRes) {
    
    pybind11::buffer_info matrBufInfo = npMat.request();
    pybind11::buffer_info vectBufInfo = npVec.request();
    pybind11::buffer_info reslBufInfo = npRes.request();

    double *matrix  = static_cast<double *>(matrBufInfo.ptr);
    double *vector  = static_cast<double *>(vectBufInfo.ptr);
    double *results = static_cast<double *>(reslBufInfo.ptr);

	#pragma omp parallel for schedule(static) num_threads(8)
	for (size_t i = 0; i < m; i++) {
		// Dot product, 4 double alla volta
		__m256d sum_vec = _mm256_set_pd(0.0, 0.0, 0.0, 0.0);
		for(size_t j = 0; j < (n/4)*4; j+=4) {
			__m256d x = _mm256_loadu_pd(matrix + i*n + j);
			__m256d y = _mm256_loadu_pd(vector + j);
			__m256d z = _mm256_mul_pd(x,y);
			sum_vec = _mm256_add_pd(sum_vec, z);
		}
		// resto + accumulo dentro result[i]
		results[i] = 0.0;
		for(int j = n-n%4; j < n; ++j)
			results[i] += matrix[i*n + j] * vector[j];
		results[i] += (sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3]);
	}
	return 0;
}

void slicedDot_forCols(pybind11::detail::unchecked_reference<float, 2l> & input,
		pybind11::detail::unchecked_mutable_reference<float, 2l> & output,
		std::vector<std::string> & tc, std::map<std::string, float> & d_rev,
		pybind11::ssize_t nrows, pybind11::ssize_t startCol, pybind11::ssize_t endCol, size_t thrIdx) {

	size_t currRow, codeIdx, currCodeLen;
	std::string currCode;
	std::map<std::string, float>::iterator it;
	std::map<std::string, float>::iterator finalDict = d_rev.end();

	for (size_t i = startCol; i < endCol; i++) {
		currRow = codeIdx = 0;
		currCode.clear();
		currCodeLen = tc[i].size();
		while (codeIdx < currCodeLen) {
			currCode.push_back(tc[i][codeIdx]);
			it = d_rev.find(currCode);
			if (it != finalDict) {
				float current_d = it->second;
				for (size_t j = 0; j < nrows; j++)
					output(j,i) += (input(j,currRow)*current_d);
				currRow++;
				currCode.clear();
			}
			codeIdx++;
		}
	}
};

void sparseSlicedDot_forCols(pybind11::detail::unchecked_reference<float, 2l> & input,
		pybind11::detail::unchecked_mutable_reference<float, 2l> & output,
		std::vector<std::string> & tc, std::map<std::string, float> & d_rev,
		std::vector<std::vector<pybind11::ssize_t>> & list_rows,
		pybind11::ssize_t nrows, pybind11::ssize_t startCol, pybind11::ssize_t endCol, size_t thrIdx) {

	size_t currRow, codeIdx, currCodeLen;
	std::string currCode;
	std::map<std::string, float>::iterator it;
	std::map<std::string, float>::iterator finalDict = d_rev.end();

	for (size_t i = startCol; i < endCol; i++) {
		currRow = codeIdx = 0;
		currCode.clear();
		currCodeLen = tc[i].size();
		while (codeIdx < currCodeLen) {
			currCode.push_back(tc[i][codeIdx]);
			it = d_rev.find(currCode);
			if (it != finalDict) {
				float current_d = it->second;
				if (current_d != 0) {
					for (size_t j = 0; j < nrows; j++)
						output(j,i) += (input(j,list_rows[i][currRow])*current_d);
				}
				currRow++;
				currCode.clear();
			}
			codeIdx++;
		}
	}
};

/*
void sparseSlicedDot_forCols_opt(pybind11::detail::unchecked_reference<float, 2l> & input,
		pybind11::detail::unchecked_mutable_reference<float, 2l> & output,
		const pybind11::list & tc, const pybind11::dict &  d_rev,
		//std::vector<std::vector<pybind11::ssize_t>> & list_rows,
		const pybind11::list & list_rows,
		pybind11::ssize_t nrows, pybind11::ssize_t startCol, pybind11::ssize_t endCol, size_t thrIdx) {

	size_t currRow, codeIdx, currCodeLen;
	std::string currCode;

	for (size_t i = startCol; i < endCol; i++) {
		currRow = codeIdx = 0;
		currCode.clear();
		auto tc_i = pybind11::cast<std::string>(*tc[i]);
		currCodeLen = tc_i.size();
		while (codeIdx < currCodeLen) {
			currCode.push_back(tc_i[codeIdx]);
			auto tmpCode = currCode.c_str();
			if (d_rev.contains(tmpCode)) {
				float current_d = pybind11::cast<float>(*(d_rev[tmpCode]));
				if (current_d != 0) {
					auto tmp1 = *(list_rows[i]);
					//size_t tmp2 = tmp1[currRow];
					for (size_t j = 0; j < nrows; j++)
						output(j,i) += 0;//(input(j,pybind11::cast<size_t>(tmp2))*current_d);
				}
				currRow++;
				currCode.clear();
			}
			codeIdx++;
		}
	}
};
*/

void slicedDot_forCols_opt(pybind11::detail::unchecked_reference<float, 2l> & input,
		pybind11::detail::unchecked_mutable_reference<float, 2l> & output,
		const pybind11::list & tc, const pybind11::dict & d_rev,
		pybind11::ssize_t nrows, pybind11::ssize_t startCol, pybind11::ssize_t endCol, size_t thrIdx) {

	size_t currRow, codeIdx, currCodeLen;
	std::string currCode;

	for (size_t i = startCol; i < endCol; i++) {
		currRow = codeIdx = 0;
		currCode.clear();
		auto tc_i = pybind11::cast<std::string>(*tc[i]);
		currCodeLen = tc_i.size();
		while (codeIdx < currCodeLen) {
			currCode.push_back(tc_i[codeIdx]);
			auto tmpCode = currCode.c_str();
			if (d_rev.contains(tmpCode)) {
				float current_d = pybind11::cast<float>(*(d_rev[tmpCode]));
				for (size_t j = 0; j < nrows; j++)
						output(j,i) += (input(j,currRow)*current_d);
				currRow++;
				currCode.clear();
			}
			codeIdx++;
		}
	}
};



void sparseSlicedDot_forCols_new(pybind11::detail::unchecked_reference<float, 2l> & input,
		pybind11::detail::unchecked_mutable_reference<float, 2l> & output,
		std::vector<std::string> & tc, std::map<std::string, float> & d_rev,
		std::vector<pybind11::ssize_t> & new_list_rows,
		std::vector<pybind11::ssize_t> & cumul_c,
		pybind11::ssize_t nrows, pybind11::ssize_t startCol, pybind11::ssize_t endCol, size_t thrIdx) {

	size_t currRow, codeIdx, currCodeLen;
	std::string currCode;
	std::map<std::string, float>::iterator it;
	std::map<std::string, float>::iterator finalDict = d_rev.end();

	for (size_t i = startCol; i < endCol; i++) {
		currRow = codeIdx = 0;
		currCode.clear();
		currCodeLen = tc[i].size();
		while (codeIdx < currCodeLen) {
			currCode.push_back(tc[i][codeIdx]);
			it = d_rev.find(currCode);
			if (it != finalDict) {
				float current_d = it->second;
				if (current_d != 0) {
					for (size_t j = 0; j < nrows; j++)
						output(j,i) += (input(j,new_list_rows[cumul_c[i]+currRow])*current_d);
				}
				currRow++;
				currCode.clear();
			}
			codeIdx++;
		}
	}
};


PYBIND11_MODULE(libmegaDot, m) {
    m.doc() = "Optimized dot product library for HAM and sHAM";
	m.def("printAMessage", &printAMessage, "Print a message, bruh");
    m.def("cpp_matDot_avx2", &cpp_matDot_avx2, "Optimized AVX2 matDot in C++");
    m.def("cpp_matDot_avx2_openmp4", &cpp_matDot_avx2_openmp4, "Optimized AVX2 matDot and with OpenMP in C++");

	m.def("dotp_cpp", [](pybind11::array_t<float> inp, std::vector<std::string> tc, std::map<std::string, float> d_rev, size_t nThr) {

		pybind11::ssize_t nrows = inp.shape(0);
		pybind11::ssize_t ncols = (pybind11::ssize_t) tc.size();
		pybind11::array_t<float, pybind11::array::f_style> outp({ nrows, ncols });
		auto output = outp.mutable_unchecked<2>();
		auto input  = inp.unchecked<2>();

		// Clear output matrix
		for (size_t i = 0; i < output.shape(1); i++) {
			for (size_t j = 0; j < output.shape(0); j++) {
				output(j,i) = 0;
			}
		}

		std::vector<size_t> colsIdx(nThr, 0);
		std::vector<size_t> cumulColsIdx(nThr + 1, 0);
		for (size_t kk = 0; kk < nThr; kk++)
			colsIdx[kk] = ncols/nThr + (kk < (ncols%nThr));
		for (size_t kk = 0; kk < nThr; kk++)
	    	cumulColsIdx[kk+1] = cumulColsIdx[kk] + colsIdx[kk];

		pybind11::gil_scoped_release release; // Ma siamo sicuri serva a qualcosa?
		//Py_BEGIN_ALLOW_THREADS
			std::vector<std::thread> threadVect;
			for (size_t i = 0; i < nThr; i++) {
				threadVect.push_back(std::thread(slicedDot_forCols, std::ref(input), std::ref(output),
						std::ref(tc), std::ref(d_rev), nrows, cumulColsIdx[i], cumulColsIdx[i+1], i));
			}
			for (size_t i = 0; i < nThr; i++)
				threadVect[i].join();
		//Py_END_ALLOW_THREADS
		pybind11::gil_scoped_acquire acquire;

		return outp;
	}, "Dot in C++, threads per colonne");

	m.def("dotp_cpp_sparse", [](std::vector<std::vector<pybind11::ssize_t>> list_rows, 
			std::vector<std::string> tc, std::map<std::string, float> d_rev, 
			pybind11::array_t<float> inp, size_t nThr) {

		pybind11::ssize_t nrows = inp.shape(0);
		pybind11::ssize_t ncols = (pybind11::ssize_t) tc.size();
		pybind11::array_t<float, pybind11::array::f_style> outp({ nrows, ncols });
		auto output = outp.mutable_unchecked<2>();
		auto input  = inp.unchecked<2>();

		// Clear output matrix
		for (size_t i = 0; i < output.shape(1); i++) {
			for (size_t j = 0; j < output.shape(0); j++) {
				output(j,i) = 0;
			}
		}

		std::vector<size_t> colsIdx(nThr, 0);
		std::vector<size_t> cumulColsIdx(nThr + 1, 0);
		for (size_t kk = 0; kk < nThr; kk++)
			colsIdx[kk] = ncols/nThr + (kk < (ncols%nThr));
		for (size_t kk = 0; kk < nThr; kk++)
	    	cumulColsIdx[kk+1] = cumulColsIdx[kk] + colsIdx[kk];

		pybind11::gil_scoped_release release; // Ma siamo sicuri serva a qualcosa?
		// Py_BEGIN_ALLOW_THREADS
			std::vector<std::thread> threadVect;
			for (size_t i = 0; i < nThr; i++) {
				threadVect.push_back(std::thread(sparseSlicedDot_forCols, std::ref(input), std::ref(output),
						std::ref(tc), std::ref(d_rev), std::ref(list_rows), nrows, cumulColsIdx[i], cumulColsIdx[i+1], i));
			}
			for (size_t i = 0; i < nThr; i++)
				threadVect[i].join();
		// Py_END_ALLOW_THREADS
		pybind11::gil_scoped_acquire acquire;

		return outp;
	}, "Dot in C++ sparsa, multithread per colonne");


	m.def("dotp_cpp_opt", [](const pybind11::array_t<float> & inp, const pybind11::list & tc, const pybind11::dict & d_rev, size_t nThr) {
		pybind11::ssize_t nrows = inp.shape(0);
		pybind11::ssize_t ncols = (pybind11::ssize_t) tc.size();
		pybind11::array_t<float, pybind11::array::f_style> outp({ nrows, ncols });
		auto output = outp.mutable_unchecked<2>();
		auto input  = inp.unchecked<2>();

		// Clear output matrix
		for (size_t i = 0; i < output.shape(1); i++) {
			for (size_t j = 0; j < output.shape(0); j++) {
				output(j,i) = 0;
			}
		}

		// WARNING: Per ora fisso il numero di thread a 1 per evitare segFault
		nThr = 1;

		std::vector<size_t> colsIdx(nThr, 0);
		std::vector<size_t> cumulColsIdx(nThr + 1, 0);
		for (size_t kk = 0; kk < nThr; kk++)
			colsIdx[kk] = ncols/nThr + (kk < (ncols%nThr));
		for (size_t kk = 0; kk < nThr; kk++)
	    	cumulColsIdx[kk+1] = cumulColsIdx[kk] + colsIdx[kk];

		//pybind11::gil_scoped_release release; // Ma siamo sicuri serva a qualcosa?
		//Py_BEGIN_ALLOW_THREADS
			std::vector<std::thread> threadVect;
			for (size_t i = 0; i < nThr; i++) {
				threadVect.push_back(std::thread(slicedDot_forCols_opt, std::ref(input), std::ref(output),
						std::ref(tc), std::ref(d_rev), nrows, cumulColsIdx[i], cumulColsIdx[i+1], i));
			}
			for (size_t i = 0; i < nThr; i++)
				threadVect[i].join();
		//Py_END_ALLOW_THREADS
		//pybind11::gil_scoped_acquire acquire;

		return outp;
	}, "Dot in C++, threads per colonne");

	m.def("dotp_cpp_sparse_new", [](std::vector<pybind11::ssize_t> new_list_rows, std::vector<pybind11::ssize_t> cumul_c,
			std::vector<std::string> tc, std::map<std::string, float> d_rev, 
			pybind11::array_t<float> inp, size_t nThr) {

		pybind11::ssize_t nrows = inp.shape(0);
		pybind11::ssize_t ncols = (pybind11::ssize_t) tc.size();
		pybind11::array_t<float, pybind11::array::f_style> outp({ nrows, ncols });
		auto output = outp.mutable_unchecked<2>();
		auto input  = inp.unchecked<2>();

		// Clear output matrix
		for (size_t i = 0; i < output.shape(1); i++) {
			for (size_t j = 0; j < output.shape(0); j++) {
				output(j,i) = 0;
			}
		}

		std::vector<size_t> colsIdx(nThr, 0);
		std::vector<size_t> cumulColsIdx(nThr + 1, 0);
		for (size_t kk = 0; kk < nThr; kk++)
			colsIdx[kk] = ncols/nThr + (kk < (ncols%nThr));
		for (size_t kk = 0; kk < nThr; kk++)
	    	cumulColsIdx[kk+1] = cumulColsIdx[kk] + colsIdx[kk];

		pybind11::gil_scoped_release release; // Ma siamo sicuri serva a qualcosa?
		// Py_BEGIN_ALLOW_THREADS
			std::vector<std::thread> threadVect;
			for (size_t i = 0; i < nThr; i++) {
				threadVect.push_back(std::thread(sparseSlicedDot_forCols_new, std::ref(input), std::ref(output),
						std::ref(tc), std::ref(d_rev), std::ref(new_list_rows), std::ref(cumul_c), nrows, cumulColsIdx[i], cumulColsIdx[i+1], i));
			}
			for (size_t i = 0; i < nThr; i++)
				threadVect[i].join();
		// Py_END_ALLOW_THREADS
		pybind11::gil_scoped_acquire acquire;

		return outp;
	}, "Dot in C++ sparsa, multithread per colonne");


	m.def("print_dict", [](const pybind11::dict & my_dict, const pybind11::list & mylist) {
		size_t dimm = mylist.size();
		for (size_t i = 0; i < dimm; i++)
			std::cout << "--- " << *(mylist[i]) << std::endl;

		std::cout << pybind11::cast<std::string>(*mylist[0]).size() << std::endl; 

		if (my_dict.contains("abc"))
			std::cout << *(my_dict["abc"]) << std::endl;
		for (auto item : my_dict)
			std::cout << item.first << " -> " << item.second << std::endl;

	}, "printDict");
}