#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>

// gcc -fPIC -shared -o hamlib.so HAMDotCver.c -pedantic -Wall -pthread

// VARIABILI GLOBALI
	
unsigned char WORDSIZE;
unsigned short K, FCL_LENGTH;
unsigned int CHAM_DIM, MATROW, MATCOL, LEFT_ROWS;
unsigned short int LMAX, BUFFDIM;
unsigned int* FIRST_CODE_L, *CHAM, *COL_END;
unsigned short int* FIRST_SYMBOL;
float * SYMBOLS;
float * RESULT;
float * VET, *LEFT_MAT;
unsigned char THREAD_COUNT;

// method to be executed by individual thread for vector-matrix dot
void* par_mat_vec(void* id){
	unsigned char my_id = (unsigned char)id;
	unsigned int start, read_weights=0, buff=0, buff_tmp=0, rem_buff=0;
	unsigned short int curlen;
	float weight;
	unsigned char bit_to_read, curr_bits=0, diff;
	char ini_offset = 0;
	const unsigned int local_m = MATCOL/THREAD_COUNT;
	unsigned int first_col = my_id*local_m;
	unsigned int last_col = (my_id + 1)*local_m -1;
	unsigned int where[local_m], ncols, HAM_word, nread_HAM_words;
	where[0]=0;
   unsigned register i, j; 

		
	if (my_id == THREAD_COUNT - 1)
		last_col = MATCOL - 1;
	ncols = 	last_col-first_col + 1;
	// the beginning of column 0 is always 0. Thus we start next loop skipping computing the
	//  beginning of the first column assigned to this thread, when thread ID is 0, since where is initialized to 0.
	start = first_col == 0 ? 1:0;
	for (i=start; i < ncols; i++){
	// where[i] contains the index in HAM vector when column i starts
		where[i] = (unsigned int)(COL_END[first_col+i-1]+1)/WORDSIZE;
//		printf("\t Thread %d -- COL_END[first_col+i-1]+1: %d,   where[%d] = %d\n", my_id, COL_END[first_col+i-1]+1, i, where[i]);
	}

   for (i=0; i < ncols; i++){
		nread_HAM_words = 0;
		if (i == 0 && first_col == 0) {
			start = 0;
			ini_offset = 0;
		}else {
			start = COL_END[i+first_col-1]+1;
			// the offset within the first HAM word of this column
			ini_offset = start - WORDSIZE*where[i]-1;
		}  		
   		HAM_word = CHAM[where[i]];
   		while (read_weights < MATROW){
			// rem_buff is the possible portion of the buffer already read but not used. start-WORDSIZE*where[i] is the possible number of bits in HAM[where[i]] that belonged to   
				bit_to_read = BUFFDIM - rem_buff;
					// completed one HAM word, need to read the next one, and account for codewords split on two HAM words
				if (bit_to_read + ini_offset + curr_bits + rem_buff> WORDSIZE){ 				
					diff = WORDSIZE - (ini_offset + curr_bits + rem_buff);
					if (rem_buff > 0)
					// shift fo most significant positions the bits remained in buff
						buff = (unsigned int)(buff << (BUFFDIM - rem_buff)) & (unsigned int)((2 << (BUFFDIM-1))-1);
					else buff = 0;			
					if(diff > 0){
					// read the remaining bits in the previous word not read
						buff_tmp = (HAM_word << ((ini_offset + curr_bits + rem_buff)) >> (ini_offset + curr_bits + rem_buff)) & (unsigned int)((2 << (BUFFDIM-1))-1);
						// moving them to most significant positions exluded those for bits previously remained in buff
						buff_tmp = (unsigned int)(buff_tmp << (BUFFDIM - rem_buff - diff)) & (unsigned int)((2 << (BUFFDIM-1))-1);
						buff += buff_tmp;
					}
					// nread_HAM_words is the number of words in HAM vector already read for this column
					nread_HAM_words++;
					HAM_word = CHAM[where[i] + nread_HAM_words];
					// left shit on buff is to consider that the bits we read from the previous HAM word are most significant bits
					buff +=  (HAM_word >> (WORDSIZE - (BUFFDIM - diff - rem_buff))) & (unsigned int)((2 << (BUFFDIM-1))-1);
					curr_bits = 0;
					ini_offset = -(diff + rem_buff); // to remind that the scanning included diff + rem_buff bits of the previous word 				
				}else{
					// left shift on HAM_word to remove possibly offeset bits at beginning (if column does not start exactly at the beginning of HAM_word) or already read bits
					// left shift on buff to move to most significanf digits the bits already read and remained in buff after the last codeword decoded. Mask with & here is to avoid buffer is extended to more that 1 byte (shift not circular)
					buff = (unsigned int)(buff << curlen) & (unsigned int)((2 << (BUFFDIM-1))-1);	
					buff += ((HAM_word << (ini_offset + curr_bits + rem_buff)) >> (WORDSIZE - bit_to_read));
				}		   
				for (j=0; j < LMAX+1; j++){
						if(FIRST_CODE_L[j] <= buff && buff < FIRST_CODE_L[j+1]){
							curlen=j;
							break;
						}
				}		   	
				// overall number of bits read so far in this HAM integer
				 curr_bits += curlen;	
				 weight = SYMBOLS[FIRST_SYMBOL[curlen] + ((buff - FIRST_CODE_L[curlen]) >> (LMAX - curlen))];
				 			
				 RESULT[i+first_col] += weight*VET[read_weights];
				 read_weights++;		   
				 rem_buff = LMAX - curlen;			   
		} 
		read_weights = 0; 
		rem_buff = 0;
		curlen = 0;
		curr_bits = 0;
		buff = 0;
	} 
	return NULL;	
}



// Function to be called from Python code and that run threads. It Computes vet-matrix dot when matrix is in the HAM format
void dot(unsigned short k, unsigned short fcl_length, unsigned int matrow, 
				unsigned int matcol, unsigned int cham_dim, unsigned short int lmax,
				unsigned char thread_count, unsigned char wordsize, 
				unsigned int* first_code_l, unsigned int* cham, 
				unsigned short int* first_symbol, unsigned int* col_end, 
				float * symbols, float* res, float * vet){
	unsigned register thread; 
	
	// INITIALIZATION OF GLOBAL VARIABLES
	THREAD_COUNT = thread_count;
	WORDSIZE = wordsize;
	LMAX = lmax;
	BUFFDIM = LMAX;
	VET = vet;
	K = k;
	FCL_LENGTH = fcl_length;
	CHAM_DIM = cham_dim;
	MATROW = matrow;
	MATCOL = matcol;
	FIRST_CODE_L = first_code_l;
	CHAM = cham;
	COL_END = col_end;
	FIRST_SYMBOL = first_symbol;
	SYMBOLS = symbols;
	RESULT = res;	


	pthread_t *thread_handles;	
	thread_handles = malloc(THREAD_COUNT*sizeof(pthread_t));

    // running  threads
	for (thread=0; thread < THREAD_COUNT; thread++)
		pthread_create(&thread_handles[thread], NULL, par_mat_vec, (void*) thread);
	// waiting for threads to finish 
	for (thread=0; thread < THREAD_COUNT; thread++)
		pthread_join(thread_handles[thread], NULL);
	
	free(thread_handles);	
		
}






// method to be executed by individual thread for matrix-matrix dot
void* par_dot_mat(void* id){
	unsigned char my_id = (unsigned char)id;
	unsigned int start, read_weights=0, buff=0, buff_tmp=0, rem_buff=0;
	unsigned short int curlen;
	float weight;
	unsigned char bit_to_read, curr_bits=0, diff;
	char ini_offset = 0;
	const unsigned int local_m = MATCOL/THREAD_COUNT;
	unsigned int first_col = my_id*local_m;
	unsigned int last_col = (my_id + 1)*local_m -1;
	unsigned int where[local_m], ncols, HAM_word, nread_HAM_words;
	where[0]=0;
   unsigned register i, j; 

		
	if (my_id == THREAD_COUNT - 1)
		last_col = MATCOL - 1;
	ncols = 	last_col-first_col + 1;
	// the beginning of column 0 is always 0. Thus we start next loop skipping computing the
	//  beginning of the first column assigned to this thread, when thread ID is 0, since where is initialized to 0.
	start = first_col == 0 ? 1:0;
	for (i=start; i < ncols; i++){
	// where[i] contains the index in HAM vector when column i starts
		where[i] = (unsigned int)(COL_END[first_col+i-1]+1)/WORDSIZE;
//		printf("\t Thread %d -- COL_END[first_col+i-1]+1: %d,   where[%d] = %d\n", my_id, COL_END[first_col+i-1]+1, i, where[i]);
	}

   for (i=0; i < ncols; i++){
		nread_HAM_words = 0;
		if (i == 0 && first_col == 0) {
			start = 0;
			ini_offset = 0;
		}else {
			start = COL_END[i+first_col-1]+1;
			// the offset within the first HAM word of this column
			ini_offset = start - WORDSIZE*where[i]-1;
		}  		
   		HAM_word = CHAM[where[i]];
   		while (read_weights < MATROW){
			// rem_buff is the possible portion of the buffer already read but not used. start-WORDSIZE*where[i] is the possible number of bits in HAM[where[i]] that belonged to   
				bit_to_read = BUFFDIM - rem_buff;
					// completed one HAM word, need to read the next one, and account for codewords split on two HAM words
				if (bit_to_read + ini_offset + curr_bits + rem_buff> WORDSIZE){ 				
					diff = WORDSIZE - (ini_offset + curr_bits + rem_buff);
					if (rem_buff > 0)
					// shift fo most significant positions the bits remained in buff
						buff = (unsigned int)(buff << (BUFFDIM - rem_buff)) & (unsigned int)((2 << (BUFFDIM-1))-1);
					else buff = 0;			
					if(diff > 0){
					// read the remaining bits in the previous word not read
						buff_tmp = (HAM_word << ((ini_offset + curr_bits + rem_buff)) >> (ini_offset + curr_bits + rem_buff)) & (unsigned int)((2 << (BUFFDIM-1))-1);
						// moving them to most significant positions exluded those for bits previously remained in buff
						buff_tmp = (unsigned int)(buff_tmp << (BUFFDIM - rem_buff - diff)) & (unsigned int)((2 << (BUFFDIM-1))-1);
						buff += buff_tmp;
					}
					// nread_HAM_words is the number of words in HAM vector already read for this column
					nread_HAM_words++;
					HAM_word = CHAM[where[i] + nread_HAM_words];
					// left shit on buff is to consider that the bits we read from the previous HAM word are most significant bits
					buff +=  (HAM_word >> (WORDSIZE - (BUFFDIM - diff - rem_buff))) & (unsigned int)((2 << (BUFFDIM-1))-1);
					curr_bits = 0;
					ini_offset = -(diff + rem_buff); // to remind that the scanning included diff + rem_buff bits of the previous word 				
				}else{
					// left shift on HAM_word to remove possibly offeset bits at beginning (if column does not start exactly at the beginning of HAM_word) or already read bits
					// left shift on buff to move to most significanf digits the bits already read and remained in buff after the last codeword decoded. Mask with & here is to avoid buffer is extended to more that 1 byte (shift not circular)
					buff = (unsigned int)(buff << curlen) & (unsigned int)((2 << (BUFFDIM-1))-1);	
					buff += ((HAM_word << (ini_offset + curr_bits + rem_buff)) >> (WORDSIZE - bit_to_read));
				}		   
				for (j=0; j < LMAX+1; j++){
						if(FIRST_CODE_L[j] <= buff && buff < FIRST_CODE_L[j+1]){
							curlen=j;
							break;
						}
				}		   	
				// overall number of bits read so far in this HAM integer
				 curr_bits += curlen;	
				 weight = SYMBOLS[FIRST_SYMBOL[curlen] + ((buff - FIRST_CODE_L[curlen]) >> (LMAX - curlen))];
				 for (j=0; j < LEFT_ROWS; j++)			
				 	RESULT[(i+first_col)*LEFT_ROWS + j] += weight*LEFT_MAT[read_weights*LEFT_ROWS + j];
				 			
				 read_weights++;		   
				 rem_buff = LMAX - curlen;			   
		} 
		read_weights = 0; 
		rem_buff = 0;
		curlen = 0;
		curr_bits = 0;
		buff = 0;
	} 
	return NULL;	
}




// Function to be called from Python code and that run threads. It Computes matrix-matrix dot when the right matrix is in the HAM format
void dotMAT(unsigned short k, unsigned short fcl_length, unsigned int matrow, 
				unsigned int matcol, unsigned int cham_dim, unsigned short int lmax,
				unsigned char thread_count, unsigned char wordsize, 
				unsigned int left_rows, // number of rows of the left matrix 
				unsigned int* first_code_l, unsigned int* cham, 
				unsigned short int* first_symbol, unsigned int* col_end, 
				float * symbols, float* res, float * left_mat){
	unsigned register thread; 
	
	// INITIALIZATION OF GLOBAL VARIABLES
	THREAD_COUNT = thread_count;
	WORDSIZE = wordsize;
	LMAX = lmax;
	BUFFDIM = LMAX;
	LEFT_MAT = left_mat;
	K = k;
	FCL_LENGTH = fcl_length;
	CHAM_DIM = cham_dim;
	MATROW = matrow;
	MATCOL = matcol;
	LEFT_ROWS = left_rows;

	FIRST_CODE_L = first_code_l;
	CHAM = cham;
	COL_END = col_end;
	FIRST_SYMBOL = first_symbol;
	SYMBOLS = symbols;
	RESULT = res;	


	pthread_t *thread_handles;	
	thread_handles = malloc(THREAD_COUNT*sizeof(pthread_t));

    // running  threads
	for (thread=0; thread < THREAD_COUNT; thread++)
		pthread_create(&thread_handles[thread], NULL, par_dot_mat, (void*) thread);
	// waiting for threads to finish 
	for (thread=0; thread < THREAD_COUNT; thread++)
		pthread_join(thread_handles[thread], NULL);
	
	free(thread_handles);	
		
}	
