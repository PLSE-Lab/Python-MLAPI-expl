#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#All imports here.
import array
import time
from multiprocessing import Process
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from math import log
from tqdm import tqdm


# # 1. Byte Files

# **Features involve :-**
# * Byte files count vectors.
# * Entropy of every file.
# * Byte files size.

# **Example data point of .byte file**
# 
# <pre>
# 00401000 00 00 80 40 40 28 00 1C 02 42 00 C4 00 20 04 20
# 00401010 00 00 20 09 2A 02 00 00 00 00 8E 10 41 0A 21 01
# 00401020 40 00 02 01 00 90 21 00 32 40 00 1C 01 40 C8 18
# 00401030 40 82 02 63 20 00 00 09 10 01 02 21 00 82 00 04
# 00401040 82 20 08 83 00 08 00 00 00 00 02 00 60 80 10 80
# 00401050 18 00 00 20 A9 00 00 00 00 04 04 78 01 02 70 90
# 00401060 00 02 00 08 20 12 00 00 00 40 10 00 80 00 40 19
# 00401070 00 00 00 00 11 20 80 04 80 10 00 20 00 00 25 00
# 00401080 00 00 01 00 00 04 00 10 02 C1 80 80 00 20 20 00
# 00401090 08 A0 01 01 44 28 00 00 08 10 20 00 02 08 00 00
# 004010A0 00 40 00 00 00 34 40 40 00 04 00 08 80 08 00 08
# 004010B0 10 00 40 00 68 02 40 04 E1 00 28 14 00 08 20 0A
# 004010C0 06 01 02 00 40 00 00 00 00 00 00 20 00 02 00 04
# 004010D0 80 18 90 00 00 10 A0 00 45 09 00 10 04 40 44 82
# 004010E0 90 00 26 10 00 00 04 00 82 00 00 00 20 40 00 00
# 004010F0 B4 00 00 40 00 02 20 25 08 00 00 00 00 00 00 00
# 00401100 08 00 00 50 00 08 40 50 00 02 06 22 08 85 30 00
# 00401110 00 80 00 80 60 00 09 00 04 20 00 00 00 00 00 00
# 00401120 00 82 40 02 00 11 46 01 4A 01 8C 01 E6 00 86 10
# 00401130 4C 01 22 00 64 00 AE 01 EA 01 2A 11 E8 10 26 11
# 00401140 4E 11 8E 11 C2 00 6C 00 0C 11 60 01 CA 00 62 10
# 00401150 6C 01 A0 11 CE 10 2C 11 4E 10 8C 00 CE 01 AE 01
# 00401160 6C 10 6C 11 A2 01 AE 00 46 11 EE 10 22 00 A8 00
# 00401170 EC 01 08 11 A2 01 AE 10 6C 00 6E 00 AC 11 8C 00
# 00401180 EC 01 2A 10 2A 01 AE 00 40 00 C8 10 48 01 4E 11
# 00401190 0E 00 EC 11 24 10 4A 10 04 01 C8 11 E6 01 C2 00
# 
# </pre>

#  ## 1.1 Byte files count vectors

# In[ ]:


#Byte file count vectors.

"""
byte_count_vectors = open("./byte_count_vectors.csv", "w+") # w+ idicates it will create a file if it does not exist in library.

char_list = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
final_string = "Id,"
for i in char_list:
    for j in char_list:
        concat = ""
        concat = concat.join((i,j))
        final_string = final_string + concat + ","

final_string = final_string+"??"
byte_count_vectors.write(final_string)
byte_count_vectors.write("\n")

files = os.listdir("./byte_files") #byte_files is folder in the local machine having all byte files.
feature_matrix = np.zeros((len(files), 257), dtype = int)

k=0 #Denotes each row. Rows will be total datapoints,here files.
print("The code is still running--------------------------------------------------------------------->>>>>>>>>>>")
for file in tqdm(files):
    f_name = file.split(".")[0]
    byte_count_vectors.write(f_name+",") #This goes into byte_fe_results.csv file.
    with open("./byte_files/"+file, "r") as each_file:
        for line in each_file:
            line = line.rstrip().split(" ") #At the end of each line there is a new space.
            line = line[1:] #Ignored the addresses.
            for hex_word in line:
                if hex_word == "??":
                    feature_matrix[k][256] += 1
                else:
                     feature_matrix[k][int(hex_word, 16)] += 1 #int(hex_word, 16) will return decimal equivalent.
    string_for_each_file = ""
    for i in feature_matrix[k]:
        string_for_each_file = string_for_each_file + str(i) + ","
    string_for_each_file = string_for_each_file[:-1]
    byte_count_vectors.write(string_for_each_file)
    byte_count_vectors.write("\n")
    each_file.close()
    k += 1

byte_count_vectors.close()
print("The code has executed----------------------------------------------------------------------------->>>>>>>>>>>>")

"""


# ## 1.2 Byte file size

# In[ ]:


#Byte file size.

"""

def calculate_byte_file_size(class_labels):
#It calculate size of each file in MB.

 fname = [] #Stores the name of each file.
 fsize = [] #Stores the size of each file respectively.
 flabels = [] #Stores respective labels of malware

 for file in tqdm(os.listdir("./byte_files")):
  file_name = file.split('.')[0]
  fname.append(file_name)
  size_in_mb = (os.stat("./byte_files/"+file).st_size)/(1024.0*1024.0)
  fsize.append(size_in_mb)
  flabels.append(int(class_labels[class_labels["Id"] == file_name]["Class"]))
 
 file_size_df = pd.DataFrame({"Class":flabels, "Id": fname, "fsize": fsize})
 
 if not os.path.exists("./byte_file_size.csv"):
  file_size_df.to_csv("./byte_file_size.csv", index = False)
 
 print(file_size_df.head())

 print("size calculated and stored in csv")

#Call the function.
class_labels = pd.read_csv("./trainLabels.csv")
calculate_byte_file_size(class_labels)

"""

#Finally merge count vectors, file sizes and class labels. Store in a single output file, bye_count_vectors.csv


# ## 1.3 Byte files entropy

# In[ ]:


"""'
data = pd.read_csv("../input/byte_count_vectors.csv")
data = data.set_index("Id")
count_vector_data = data.drop(['Class', 'fsize'], axis=1)
count_vector_data['entropy'] = 0.0
entropies = []


for idx, rows in tqdm(count_vector_data.iterrows()):
    entropy = 0.0
    for elem in rows.index:
        if(rows[elem]):
            entropy = entropy - ( ((rows[elem]/rows.sum()) * log(rows[elem]/rows.sum())))
    
    entropies.append(entropy)

count_vector_data['entropy'] = entropies

data['entropy'] = count_vector_data['entropy']

#byte_entropy.to_csv("byte_entropy.csv")'"""


# ## 1.5 Summary on Feature Extraction of byte files

# 1. For byte file cout vectors, we have used simple bag of words to count the appeareance of each hex (eg: f3,f2,ff etc) in a data file.
# 1. For byte file size , we have calculated the size of each .byte file.
# 1. **'byte_count_vectors.csv'** contains BoW, file size in a dataframe.
# 1. **'byte_entropy.csv'** contains entropy of each file. We hvae utilized the idea of the entropy of a file. We already had our count vectors in 'byte_count_vectors.csv', from which we have calculated the entropy of each and every single file.
# 1. We could'nt utilized the idea of bi-grams, n-grams since somehow our machine failed in terms of memory.
# 1. **'byte_final_features.csv'** consist of all above data combined, i.e count_vectors,file_size,entropy ina  single file.

# # 2. ASM Files

# **Features involve :-**
# * ASM files count vectors.
# * Image features of each file.
# * ASM files size.

# ## 2.1 ASM files count vectors

# In[ ]:


#ASM file count vectors.
#We will divide asm files in 3 different folders, 'first', 'second', 'third' respectively. 
#It will helpp us in multiprogramming.
"""
def firstprocess():
    list_of_dics = []
    files  = os.listdir("./first") 
    for file in tqdm(files):
        filename = file.split('.')[0]
        prefixes = {'HEADER:': 0, '.text:': 0, '.Pav:': 0, '.idata:': 0, '.data:': 0, '.bss:': 0, '.rdata:': 0, '.edata:': 0, '.rsrc:': 0, '.tls:': 0, '.reloc:': 0, '.BSS:': 0, '.CODE': 0}
        opcode = {'add': 0, 'al': 0, 'bt': 0, 'call': 0, 'cdq': 0, 'cld': 0, 'cli': 0, 'cmc': 0, 'cmp': 0, 'const': 0, 'cwd': 0, 'daa': 0, 'db': 0, 'dd': 0,
                 'dec': 0, 'dw': 0, 'endp': 0, 'ends': 0, 'faddp': 0, 'fchs': 0, 'fdiv': 0, 'fdivp': 0, 'fdivr': 0, 'fild': 0, 'fistp': 0, 'fld': 0,
                  'fstcw': 0, 'fstcwimul': 0, 'fstp': 0, 'fword': 0, 'fxch': 0, 'imul': 0, 'in': 0, 'inc': 0, 'ins': 0, 'int': 0, 'jb': 0, 'je': 0, 'jg': 0,
                   'jge': 0, 'jl': 0, 'jmp': 0, 'jnb': 0, 'jno': 0, 'jnz': 0, 'jo': 0, 'jz': 0, 'lea': 0, 'loope': 0, 'mov': 0, 'movzx': 0, 'mul': 0,
                    'near': 0, 'neg': 0, 'not': 0, 'or': 0, 'out': 0, 'outs': 0, 'pop': 0, 'popf': 0, 'proc': 0, 'push': 0, 'pushf': 0, 'rcl': 0, 'rcr': 0,
                    'rdtsc': 0, 'rep': 0, 'ret': 0, 'retn': 0, 'rol': 0, 'ror': 0, 'sal': 0, 'sar': 0, 'sbb': 0, 'scas': 0, 'setb': 0, 'setle': 0,
                     'setnle': 0, 'setnz': 0, 'setz': 0, 'shl': 0, 'shld': 0, 'shr': 0, 'sidt': 0, 'stc': 0, 'std': 0, 'sti': 0, 'stos': 0, 'sub': 0,
                     'test': 0, 'wait': 0, 'xchg': 0, 'xor': 0, 'retf': 0, 'nop': 0, 'rtn': 0}
        #Opcodes were changed after dchad solution.
        keywords = {'.dll' : 0, 'std::' : 0, ':dword' : 0}
        registers = {'edx': 0, 'esi': 0, 'es': 0, 'fs': 0, 'ds': 0, 'ss': 0, 'gs': 0, 'cs': 0, 'ah': 0, 'al': 0, 'ax': 0, 'bh': 0, 'bl': 0, 'bx': 0,
                    'ch': 0, 'cl': 0, 'cx': 0, 'dh': 0, 'dl': 0, 'dx': 0, 'eax': 0, 'ebp': 0, 'ebx': 0, 'ecx': 0, 'edi': 0, 'esp': 0, 'eip': 0}
        #Registers were changed after dchad solution.
        current_file = open("./first/"+file, "r", encoding ="cp1252", errors = "replace")
        for lines in current_file:
            line = lines.rstrip().split()
            prefix = line[0]
            rest_of_line = line[1:]        
            #Check for prefixes
            for key in prefixes.keys():
                if key in prefix:
                    prefixes[key] += 1
            #Check for keywords
            for key in keywords.keys():
                for word in rest_of_line:
                    if key in word: #Because we need to match substring.
                        keywords[key] += 1
            #Check for opcodes
            for key in opcode.keys():
                for word in rest_of_line:
                    if key==word: #Because we need to match exact string.
                        opcode[key] += 1
            #Check for registers
            if ('text' in prefix or 'CODE' in prefix):
                for key in registers.keys():
                    for word in rest_of_line:
                        if key in word:
                            registers[key] += 1
        current_file.close()
        final_dic = {'Id': filename, }
        #final_dic['Id'] = filename
        for key,values in prefixes.items():
            final_dic[key] = values
        for key,values in keywords.items():
            final_dic[key] = values
        for key,values in opcode.items():
            final_dic[key] = values
        for key,values in registers.items():
            final_dic[key] = values
        list_of_dics.append(final_dic)
    first_df = pd.DataFrame(list_of_dics)
    first_df = first_df.set_index("Id")
    first_df.to_csv("./first/firstfile.csv")
    
def secondprocess():
    list_of_dics = []
    files  = os.listdir("./second")
    for file in tqdm(files):
        filename = file.split('.')[0]
        prefixes = {'HEADER:': 0, '.text:': 0, '.Pav:': 0, '.idata:': 0, '.data:': 0, '.bss:': 0, '.rdata:': 0, '.edata:': 0, '.rsrc:': 0, '.tls:': 0, '.reloc:': 0, '.BSS:': 0, '.CODE': 0}
        opcode = {'add': 0, 'al': 0, 'bt': 0, 'call': 0, 'cdq': 0, 'cld': 0, 'cli': 0, 'cmc': 0, 'cmp': 0, 'const': 0, 'cwd': 0, 'daa': 0, 'db': 0, 'dd': 0,
                 'dec': 0, 'dw': 0, 'endp': 0, 'ends': 0, 'faddp': 0, 'fchs': 0, 'fdiv': 0, 'fdivp': 0, 'fdivr': 0, 'fild': 0, 'fistp': 0, 'fld': 0,
                  'fstcw': 0, 'fstcwimul': 0, 'fstp': 0, 'fword': 0, 'fxch': 0, 'imul': 0, 'in': 0, 'inc': 0, 'ins': 0, 'int': 0, 'jb': 0, 'je': 0, 'jg': 0,
                   'jge': 0, 'jl': 0, 'jmp': 0, 'jnb': 0, 'jno': 0, 'jnz': 0, 'jo': 0, 'jz': 0, 'lea': 0, 'loope': 0, 'mov': 0, 'movzx': 0, 'mul': 0,
                    'near': 0, 'neg': 0, 'not': 0, 'or': 0, 'out': 0, 'outs': 0, 'pop': 0, 'popf': 0, 'proc': 0, 'push': 0, 'pushf': 0, 'rcl': 0, 'rcr': 0,
                    'rdtsc': 0, 'rep': 0, 'ret': 0, 'retn': 0, 'rol': 0, 'ror': 0, 'sal': 0, 'sar': 0, 'sbb': 0, 'scas': 0, 'setb': 0, 'setle': 0,
                     'setnle': 0, 'setnz': 0, 'setz': 0, 'shl': 0, 'shld': 0, 'shr': 0, 'sidt': 0, 'stc': 0, 'std': 0, 'sti': 0, 'stos': 0, 'sub': 0,
                     'test': 0, 'wait': 0, 'xchg': 0, 'xor': 0, 'retf': 0, 'nop': 0, 'rtn': 0}
        #Opcodes were changed after dchad solution.
        keywords = {'.dll' : 0, 'std::' : 0, ':dword' : 0}
        registers = {'edx': 0, 'esi': 0, 'es': 0, 'fs': 0, 'ds': 0, 'ss': 0, 'gs': 0, 'cs': 0, 'ah': 0, 'al': 0, 'ax': 0, 'bh': 0, 'bl': 0, 'bx': 0,
                    'ch': 0, 'cl': 0, 'cx': 0, 'dh': 0, 'dl': 0, 'dx': 0, 'eax': 0, 'ebp': 0, 'ebx': 0, 'ecx': 0, 'edi': 0, 'esp': 0, 'eip': 0}
        #Registers were changed after dchad solution.
        current_file = open("./second/"+file, "r", encoding ="cp1252", errors = "replace")
        for lines in current_file:
            line = lines.rstrip().split()
            prefix = line[0]
            rest_of_line = line[1:]    
            #Check for prefixes
            for key in prefixes.keys():
                if key in prefix:
                    prefixes[key] += 1
            #Check for keywords
            for key in keywords.keys():
                for word in rest_of_line:
                    if key in word: #Because we need to match substring.
                        keywords[key] += 1
            #Check for opcodes
            for key in opcode.keys():
                for word in rest_of_line:
                    if key==word: #Because we need to match exact string.
                        opcode[key] += 1
            #Check for registers
            if ('text' in prefix or 'CODE' in prefix):
                for key in registers.keys():
                    for word in rest_of_line:
                        if key in word:
                            registers[key] += 1
        current_file.close()
        final_dic = {'Id': filename, }
        #final_dic['Id'] = filename
        for key,values in prefixes.items():
            final_dic[key] = values
        for key,values in keywords.items():
            final_dic[key] = values
        for key,values in opcode.items():
            final_dic[key] = values
        for key,values in registers.items():
            final_dic[key] = values
        list_of_dics.append(final_dic)
    first_df = pd.DataFrame(list_of_dics)
    first_df = first_df.set_index("Id")
    first_df.to_csv("./second/secondfile.csv")

def thirdprocess():
    list_of_dics = []
    files  = os.listdir("./third")
    for file in tqdm(files):
        filename = file.split('.')[0]
        filename = file.split('.')[0]
        prefixes = {'HEADER:': 0, '.text:': 0, '.Pav:': 0, '.idata:': 0, '.data:': 0, '.bss:': 0, '.rdata:': 0, '.edata:': 0, '.rsrc:': 0, '.tls:': 0, '.reloc:': 0, '.BSS:': 0, '.CODE': 0}
        opcode = {'add': 0, 'al': 0, 'bt': 0, 'call': 0, 'cdq': 0, 'cld': 0, 'cli': 0, 'cmc': 0, 'cmp': 0, 'const': 0, 'cwd': 0, 'daa': 0, 'db': 0, 'dd': 0,
                 'dec': 0, 'dw': 0, 'endp': 0, 'ends': 0, 'faddp': 0, 'fchs': 0, 'fdiv': 0, 'fdivp': 0, 'fdivr': 0, 'fild': 0, 'fistp': 0, 'fld': 0,
                  'fstcw': 0, 'fstcwimul': 0, 'fstp': 0, 'fword': 0, 'fxch': 0, 'imul': 0, 'in': 0, 'inc': 0, 'ins': 0, 'int': 0, 'jb': 0, 'je': 0, 'jg': 0,
                   'jge': 0, 'jl': 0, 'jmp': 0, 'jnb': 0, 'jno': 0, 'jnz': 0, 'jo': 0, 'jz': 0, 'lea': 0, 'loope': 0, 'mov': 0, 'movzx': 0, 'mul': 0,
                    'near': 0, 'neg': 0, 'not': 0, 'or': 0, 'out': 0, 'outs': 0, 'pop': 0, 'popf': 0, 'proc': 0, 'push': 0, 'pushf': 0, 'rcl': 0, 'rcr': 0,
                    'rdtsc': 0, 'rep': 0, 'ret': 0, 'retn': 0, 'rol': 0, 'ror': 0, 'sal': 0, 'sar': 0, 'sbb': 0, 'scas': 0, 'setb': 0, 'setle': 0,
                     'setnle': 0, 'setnz': 0, 'setz': 0, 'shl': 0, 'shld': 0, 'shr': 0, 'sidt': 0, 'stc': 0, 'std': 0, 'sti': 0, 'stos': 0, 'sub': 0,
                     'test': 0, 'wait': 0, 'xchg': 0, 'xor': 0, 'retf': 0, 'nop': 0, 'rtn': 0}
        #Opcodes were changed after dchad solution.
        keywords = {'.dll' : 0, 'std::' : 0, ':dword' : 0}
        registers = {'edx': 0, 'esi': 0, 'es': 0, 'fs': 0, 'ds': 0, 'ss': 0, 'gs': 0, 'cs': 0, 'ah': 0, 'al': 0, 'ax': 0, 'bh': 0, 'bl': 0, 'bx': 0,
                    'ch': 0, 'cl': 0, 'cx': 0, 'dh': 0, 'dl': 0, 'dx': 0, 'eax': 0, 'ebp': 0, 'ebx': 0, 'ecx': 0, 'edi': 0, 'esp': 0, 'eip': 0}
        #Registers were changed after dchad solution.
        current_file = open("./third/"+file, "r", encoding ="cp1252", errors = "replace")
        for lines in current_file:
            line = lines.rstrip().split()
            prefix = line[0]
            rest_of_line = line[1:]        
            #Check for prefixes
            for key in prefixes.keys():
                if key in prefix:
                    prefixes[key] += 1
            #Check for keywords
            for key in keywords.keys():
                for word in rest_of_line:
                    if key in word: #Because we need to match substring.
                        keywords[key] += 1
            #Check for opcodes
            for key in opcode.keys():
                for word in rest_of_line:
                    if key==word: #Because we need to match exact string.
                        opcode[key] += 1
            #Check for registers
            if ('text' in prefix or 'CODE' in prefix):
                for key in registers.keys():
                    for word in rest_of_line:
                        if key in word:
                            registers[key] += 1
        current_file.close()
        final_dic = {'Id': filename, }
        #final_dic['Id'] = filename
        for key,values in prefixes.items():
            final_dic[key] = values
        for key,values in keywords.items():
            final_dic[key] = values
        for key,values in opcode.items():
            final_dic[key] = values
        for key,values in registers.items():
            final_dic[key] = values
        list_of_dics.append(final_dic)
    first_df = pd.DataFrame(list_of_dics)
    first_df = first_df.set_index("Id")
    first_df.to_csv("./third/thirdfile.csv")
    
def main():
    p1=Process(target=firstprocess)
    p2=Process(target=secondprocess)
    p3=Process(target=thirdprocess)
    #p4=Process(target=fourthprocess)
    #p5=Process(target=fifthprocess)
    #p1.start() is used to start the thread execution
    p1.start()
    p2.start()
    p3.start()
    #p4.start()
    #p5.start()
    #After completion all the threads are joined
    p1.join()
    p2.join()
    p3.join()
    #p4.join()
    #p5.join()

if __name__=="__main__":
    main()
"""


# ## 2.2 ASM files image features

# In[ ]:


"""
def firstprocess():
    list_of_dics = []
    files  = os.listdir("./first")
    for file in tqdm(files):
        filename = file.split('.')[0]
        current_file = open("./first/"+file, mode = "rb")
        ln = os.path.getsize("./first/"+file)
        width = int(ln**0.5)
        rem = ln%width
        a = (array.array("B"))
        a.fromfile(current_file, ln-rem)
        g = np.reshape(a, (len(a)//width, width))
        g = np.uint8(g)
        final_image_feature_array = g.flatten()[0:1000]
        current_file.close()
        keys = list("img_feature_"+ str(i) for i in range(0,1000))
        final_dic = {'Id': filename, }
        for key in keys:
            final_dic[key] =  final_image_feature_array[int(key.split('_')[2])]
        list_of_dics.append(final_dic)
	
    first_df = pd.DataFrame(list_of_dics)
    first_df = first_df.set_index("Id")
    first_df.to_csv("./first/image_feature_firstfile.csv")
    
def secondprocess():
    list_of_dics = []
    files  = os.listdir("./second")
    for file in tqdm(files):
        filename = file.split('.')[0]
        current_file = open("./second/"+file, mode = "rb")
        ln = os.path.getsize("./second/"+file)
        width = int(ln**0.5)
        rem = ln%width
        a = (array.array("B"))
        a.fromfile(current_file, ln-rem)
        g = np.reshape(a, (len(a)//width, width))
        g = np.uint8(g)
        final_image_feature_array = g.flatten()[0:1000]
        current_file.close()
        keys = list("img_feature_"+ str(i) for i in range(0,1000))
        final_dic = {'Id': filename, }
        for key in keys:
            final_dic[key] =  final_image_feature_array[int(key.split('_')[2])]
        list_of_dics.append(final_dic)
	
    first_df = pd.DataFrame(list_of_dics)
    first_df = first_df.set_index("Id")
    first_df.to_csv("./second/image_feature_secondfile.csv")

def thirdprocess():
    list_of_dics = []
    files  = os.listdir("./third")
    for file in tqdm(files):
        filename = file.split('.')[0]
        current_file = open("./third/"+file, mode = "rb")
        ln = os.path.getsize("./third/"+file)
        width = int(ln**0.5)
        rem = ln%width
        a = (array.array("B"))
        a.fromfile(current_file, ln-rem)
        g = np.reshape(a, (len(a)//width, width))
        g = np.uint8(g)
        final_image_feature_array = g.flatten()[0:1000]
        current_file.close()
        keys = list("img_feature_"+ str(i) for i in range(0,1000))
        final_dic = {'Id': filename, }
        for key in keys:
            final_dic[key] =  final_image_feature_array[int(key.split('_')[2])]
        list_of_dics.append(final_dic)
	
    first_df = pd.DataFrame(list_of_dics)
    first_df = first_df.set_index("Id")
    first_df.to_csv("./third/image_feature_thirdfile.csv")
    
def main():
    print(time.ctime())
    p1=Process(target=firstprocess)
    p2=Process(target=secondprocess)
    p3=Process(target=thirdprocess)
    #p4=Process(target=fourthprocess)
    #p5=Process(target=fifthprocess)
    #p1.start() is used to start the thread execution
    p1.start()
    p2.start()
    p3.start()
    #p4.start()
    #p5.start()
    #After completion all the threads are joined
    p1.join()
    p2.join()
    p3.join()
    print(time.ctime())
    #p4.join()
    #p5.join()

if __name__=="__main__":
    main()

"""


# 1. We divided asm files in three different folders 'first', 'second', 'third' to do multiprogramming.
# 1. Each will return a different csv files. 
# > firstfile.csv, image_feature_firstfile.csv containing count vectors and image features of all thee files of 'first' folder and so on.
# 1. **asm_count_vectors.csv** consist of count vectors of all thee asm files.
# 1. **asm_file_size.csv** consist of the file sizes of all the asm files.
# 1. **asm_final_features.csv** consist of all the count_vectors, image_features and file_sizes commbined. 

# # Final summary

# * byte_final_features have all the data extracted from byte files.
# * asm_final_features have all the data extracted from asm files.

# In[ ]:


asm_final_features = pd.read_csv("../input/asm_final_features.csv")
asm_final_features.head()


# In[ ]:


asm_final_features = asm_final_features.set_index("Id")


# In[ ]:


asm_final_features.head()


# In[ ]:


byte_final_features = pd.read_csv("../input/byte_final_features.csv")
byte_final_features.head()

