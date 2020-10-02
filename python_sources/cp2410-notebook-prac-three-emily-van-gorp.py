#!/usr/bin/env python
# coding: utf-8

# Q1. A recursive algorithm that can be used to find S of n elements is linear recursion. For an input size n, the algorithm makes n + 1 function calls. This means it will take O(n) time and since it spends all of its time performing the non-recursice part of each call, it also uses O(n) space.

# Q2. def power(2, 5)<br>
# n = 5, 2 * power(2, 5 - 1)<br>
# n = 4,  2 * power(2, 4 - 1)<br>
# n = 3,  2 * power(2, 3 - 1)<br>
# n = 2,  2 * power(2, 2 - 1)<br>
# n = 1,  2 * power(2, 1-1) <br>
# n = 0,  2 * 1<br>
# 2 * 1 = 2<br>
# 2 * 2 = 4<br>
# 4* 2 = 8<br>
# 8 * 2 = 16<br>
# 16 * 2 = 32<br>

# Q3. def power(2, 18)<br>
# n = 18, power(2, 9)<br>
# n = 9, power(2, 4)<br>
# n = 4, power(2, 2)<br>
# n = 2, power(2, 1)<br>
# n = 1, power(2, 0)<br>
# n = 0, partial = 1
# 1 * 1 = 1
# 1 + 1 * 2 = 3
# partial = 3
# result = 3 * 3 = 9
# 9 + 9 * 2 = 27
# partial = 27
# result = 27
# result = 27 * 27 = 729
# 729 + 729 * 2 = 2187
# partial = 2187
# result = 2187 * 2187 = 4374
