
# A pure-python script to find as many primes as possible in 10 seconds 

import numpy, math, sys
from time import time

def calc_primes(n):
    # Find all primes n > prime > 2 using the Sieve of Eratosthenes 
    # For efficiency, track only odd numbers (evens are nonprime)
    
    sieve = numpy.ones(n/2, dtype=numpy.bool) 
    limit = int(math.sqrt(n)) + 1 
    
    for i in range(3, limit, 2): 
        if sieve[i/2]:
            sieve[i*i/2 :: i] = False
            
    prime_indexes = numpy.nonzero(sieve)[0][1::]
    primes  = 2 * prime_indexes.astype(numpy.int32) + 1 
    return primes


def ints_tofile(ints_arr, file_name, line_chars=10):
    # This custom output function is faster than numpy's .tofile() 
    
    buf  = numpy.zeros(shape=(len(ints_arr), line_chars), dtype=numpy.int8) 
    buf[:, line_chars-1] = 10   # 10 = ASCII linefeed
    
    for buf_ix in range(line_chars-2, 0-1, -1):
        numpy.mod(ints_arr, 10, out=buf[:, buf_ix])
        buf[:, buf_ix] += 48    # 48 = ASCII '0'
        ints_arr /= 10        
        
    fout = open(file_name, "wb")
    fout.write(buf.tobytes())
    fout.close()


def timed_primes(n):
    
    t_start= time()
    primes = calc_primes(n)
    t_calc = time() - t_start
    
    # primes.tofile('primes.csv', sep=',\n') # slower
    ints_tofile(primes, 'primes.csv', line_chars=10)
    t_all  = time() - t_start
    t_save = t_all  - t_calc
    
    formatting = "Found %-10i primes less than %-10i | time: %6.3f  calc_time: %6.3f  save_time: %6.3f"
    values = len(primes), n, t_all, t_calc, t_save
    print(formatting % values)
    sys.stdout.flush()
    return t_all


def main():
    mil = 1000000
    upper_bound = 500*mil
    timed_primes(upper_bound)
    
main()
