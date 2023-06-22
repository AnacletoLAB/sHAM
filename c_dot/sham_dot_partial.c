#include<pthread.h>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
#include<string.h>

// VARIABILI GLOBALI
unsigned char WORDSIZE;
unsigned short K, FCL_LENGTH;
unsigned int CHAM_DIM, MATROW, MATCOL, LEFT_ROWS;
unsigned short int LMAX, BUFFDIM;
unsigned int* FIRST_CODE_L, *CHAM, *COL_END, *RI, *CB;

unsigned int* PAR_TABLE;
unsigned short int T;

unsigned short int* FIRST_SYMBOL;
float * SYMBOLS;
float * RESULT;
float * VET, *LEFT_MAT;
unsigned char THREAD_COUNT;

void* par_mat_vec(void* id){
	unsigned char my_id = (unsigned char)id;
	unsigned int i,j, start, read_weights=0, ri_pos=0, buff=0, buff_tmp=0, rem_buff=0;
	unsigned short int curlen;
	float weight;
//	unsigned char buff=0, buff_tmp=0, rem_buff=0, bit_to_read, curr_bits=0, diff;
	unsigned char bit_to_read, curr_bits=0, diff;

	char ini_offset = 0;
	const unsigned int local_m = MATCOL/THREAD_COUNT;
	unsigned int first_col = my_id*local_m;
	unsigned int last_col = (my_id + 1)*local_m -1;
	unsigned int where[local_m], ncols, HAM_word, nread_HAM_words;
	where[0]=0;
	
	if (my_id == THREAD_COUNT - 1)
		last_col = MATCOL - 1;
	ncols = 	last_col-first_col + 1;
	// the beginning of column 0 is always 0. Thus we start next loop skipping computing the
	//  beginning of the first column assigned to this thread, when thread ID is 0, since where is initialized to 0.
	start = first_col == 0 ? 1:0;
	for (j = 0; j<first_col;j++) ri_pos += CB[j];//finding intial position in array ri
//	printf("\t Thread %d -- ncols: %d, first_col: %d, last_col: %d \n", my_id, ncols, first_col, last_col);
	for (i=start; i < ncols; i++){
	// where[i] contains the index in HAM vector when column i starts
		where[i] = (unsigned int)(COL_END[first_col+i-1]+1)/WORDSIZE;
//		printf("\t Thread %d -- COL_END[first_col+i-1]+1: %d,   where[%d] = %d\n", my_id, COL_END[first_col+i-1]+1, i, where[i]);
	}
//	for (i=0; i < 1; i++){
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
   		while (read_weights < CB[i+first_col]){
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
						   
				// PAR_TABLE[j] is the length of the first t-sized codeword of buff, if there is one, oterwhise it tells us where in FIRST_CODE_L we have to start in order to find the actual length of next codeword
				j = PAR_TABLE[buff >> (LMAX - T)];
				while(buff >= FIRST_CODE_L[j+1]){
					j++;
				}
				curlen = j;

				// overall number of bits read so far in this HAM integer
				 curr_bits += curlen;	
				 weight = SYMBOLS[FIRST_SYMBOL[curlen] + ((buff - FIRST_CODE_L[curlen]) >> (LMAX - curlen))];		
				 RESULT[i+first_col] += weight*VET[RI[ri_pos++]];
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

// per shared object 
// COMPILE  gcc -fPIC -shared -o shamlib.so sHAMDotCver.c -pedantic -Wall -pthread

void dot(unsigned short k, unsigned short fcl_length, unsigned int matrow, 
				unsigned int matcol, unsigned int cham_dim, unsigned short int lmax,
				unsigned char thread_count, unsigned char wordsize, 
				unsigned int* first_code_l, unsigned int* cham, 
				unsigned short int* first_symbol, unsigned int* col_end, 
				float * symbols, float* res, float * vet, unsigned int *ri, unsigned int *cb, unsigned int* par_table, unsigned short int t){
	unsigned register thread; 
	srand(123);
	
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
	RI = ri;
	CB = cb;

	PAR_TABLE = par_table;
	T=t;

	pthread_t *thread_handles;
	
	thread_handles = malloc(THREAD_COUNT*sizeof(pthread_t));

	for (thread=0; thread < THREAD_COUNT; thread++)
		pthread_create(&thread_handles[thread], NULL, par_mat_vec, (void*) thread);
	
	for (thread=0; thread < THREAD_COUNT; thread++)
		pthread_join(thread_handles[thread], NULL);
		
	free(thread_handles);

}	

// function to compute the dot when on the left we have a matrix, non a vector
void* par_dot_mat(void* id){
	unsigned char my_id = (unsigned char)id;
	unsigned int i,j, start, read_weights=0, ri_pos=0, buff=0, buff_tmp=0, rem_buff=0;
	unsigned short int curlen;
	float weight;
//	unsigned char buff=0, buff_tmp=0, rem_buff=0, bit_to_read, curr_bits=0, diff;
	unsigned char bit_to_read, curr_bits=0, diff;

	char ini_offset = 0;
	const unsigned int local_m = MATCOL/THREAD_COUNT;
	unsigned int first_col = my_id*local_m;
	unsigned int last_col = (my_id + 1)*local_m -1;
	unsigned int where[local_m], ncols, HAM_word, nread_HAM_words;
	where[0]=0;
	
	if (my_id == THREAD_COUNT - 1)
		last_col = MATCOL - 1;
	ncols = 	last_col-first_col + 1;
	// the beginning of column 0 is always 0. Thus we start next loop skipping computing the
	//  beginning of the first column assigned to this thread, when thread ID is 0, since where is initialized to 0.
	start = first_col == 0 ? 1:0;
	for (j = 0; j<first_col;j++) ri_pos += CB[j];//finding intial position in array ri
//	printf("\t Thread %d -- ncols: %d, first_col: %d, last_col: %d \n", my_id, ncols, first_col, last_col);
	for (i=start; i < ncols; i++){
	// where[i] contains the index in HAM vector when column i starts
		where[i] = (unsigned int)(COL_END[first_col+i-1]+1)/WORDSIZE;
//		printf("\t Thread %d -- COL_END[first_col+i-1]+1: %d,   where[%d] = %d\n", my_id, COL_END[first_col+i-1]+1, i, where[i]);
	}
//	for (i=0; i < 1; i++){
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

   		while (read_weights < CB[i+first_col]){
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
						   
				// PAR_TABLE[j] is the length of the first t-sized codeword of buff, if there is one, oterwhise it tells us where in FIRST_CODE_L we have to start in order to find the actual length of next codeword
				j = PAR_TABLE[buff >> (LMAX - T)];
				while(buff >= FIRST_CODE_L[j+1]){
					j++;
				}
				curlen = j;

				// overall number of bits read so far in this HAM integer
				 curr_bits += curlen;	
				 weight = SYMBOLS[FIRST_SYMBOL[curlen] + ((buff - FIRST_CODE_L[curlen]) >> (LMAX - curlen))];
				 //if(i<1) 
				 //	printf("\t\t Thread %d -- i: %d, read_weights: %d,  buff %d,  curlen  %d, rem_buff: %d, ini_offset: %d,  weight %f,  ri_pos: %u\n", 
				//		my_id, i, read_weights, buff, curlen, rem_buff, ini_offset, weight, ri_pos);
				 for (j=0; j < LEFT_ROWS; j++)			
				 	RESULT[(i+first_col)*LEFT_ROWS + j] += weight*LEFT_MAT[RI[ri_pos]*LEFT_ROWS + j];
				 
				 //if(i<1) printf("\t\t dopo for RESULT[(i+first_col)*LEFT_ROWS + 0]: %f\n", RESULT[(i+first_col)*LEFT_ROWS + 0]);
				 ri_pos++;
				 read_weights++;   
				 rem_buff = LMAX - curlen;	
				
		}
//		RESULT[i+first_col] = sum;  
		read_weights = 0; 
		rem_buff = 0;
		curlen = 0;
		curr_bits = 0;
		buff = 0;
	} 
	return NULL;	
}
void dotMAT(unsigned short k, unsigned short fcl_length, unsigned int matrow, 
				unsigned int matcol, unsigned int cham_dim, unsigned short int lmax,
				unsigned char thread_count, unsigned char wordsize, unsigned int left_rows, 
				unsigned int* first_code_l, unsigned int* cham, 
				unsigned short int* first_symbol, unsigned int* col_end, 
				float * symbols, float* res, float * left_mat, unsigned int *ri, unsigned int *cb, unsigned int* par_table, unsigned short int t){
	unsigned register thread; 
	srand(123);
	
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
	RI = ri;
	CB = cb;

	PAR_TABLE = par_table;
	T=t;

	pthread_t *thread_handles;
//	for(j=0; j < 5; j++){
//		for(thread=0; thread < LEFT_ROWS; thread++)
//			printf("\t j : %d, thread: %d, LEFT_MAT[j*LEFT_ROWS + thread]: %f\n", j, thread, LEFT_MAT[j*LEFT_ROWS + thread]);
//	}		
	thread_handles = malloc(THREAD_COUNT*sizeof(pthread_t));

	for (thread=0; thread < THREAD_COUNT; thread++)
		pthread_create(&thread_handles[thread], NULL, par_dot_mat, (void*) thread);
	
	for (thread=0; thread < THREAD_COUNT; thread++)
		pthread_join(thread_handles[thread], NULL);
		
	free(thread_handles);

}
