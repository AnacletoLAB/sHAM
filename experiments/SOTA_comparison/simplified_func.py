import numpy as np


### 3: DOT PARALLEL (CHIAMATA CON STARMAP)
# inp: matrice di sx
# tc: risultato della funzione sotto
def dotp(inp, tc, d_rev):
    output = np.zeros((inp.shape[0], len(tc)), dtype='float32', order='F')
    for col, t in enumerate(tc):
        row = 0
        curr = ""
        for b in t:
            curr += b
            if curr in d_rev:
                current_d = d_rev[curr]
                if current_d != 0:
                    output = mult_for_row(inp, output, row, current_d, col)
                curr = ""
                row += 1
    return output

### 2: prende output di 1 e convert in lista di eq di bit ogni elem è una colonna
def to_convert(info, list_string_col):
    to_conv = []
    for i in range(len(info)-1):
        if info[i][0] == info[i+1][0]:
            to_conv += [list_string_col[info[i][0]][info[i][1]:info[i+1][1]]]
        else:
            if info[i][0] + 1 == info[i+1]:
                to_conv += [list_string_col[info[i][0]][info[i][1]:]+list_string_col[info[i+1][0]][:info[i+1][1]]]
            else:
                elem = ""
                elem += list_string_col[info[i][0]][info[i][1]:]
                for j in range(info[i][0]+1, info[i+1][0]):
                    elem += list_string_col[j]
                if info[i+1][1] != 0:
                    elem += list_string_col[info[i+1][0]][:info[i+1][1]]
                to_conv += [elem]
    return to_conv

### 1: rest stringhe di bi tlunghe come le parole macchina
# Info: in che posiz e parola inizia una colonna
def make_word_with_info(matr, encoded, bit_words_machine=64):
    bit = bit_words_machine
    list_string_col =[]
    string = ''
    row = 0
    info = []

    for x in np.nditer(encoded, order='F'):
        if row == 0:
            info += [(len(list_string_col), len(string))]
        string += (np.array2string(x)[1:-1])
        if len(string) > bit:
            bit_overflow = len(string)%bit
            list_string_col += [string[:-bit_overflow]]
            string = string[-bit_overflow:]
        elif len(string) == bit:
            list_string_col += [string]
            string = ''
        row += 1
        if row >= matr.shape[0]:
            row = 0

    bit_remain = len(string)
    info += [(len(list_string_col), bit_remain)]
    if bit_remain > 0:
        string += "0"*(bit-bit_remain) #padding di 0 per renderla lunga bit
        list_string_col += [string]

    return list_string_col, info