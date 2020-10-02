#!/usr/bin/env python
# coding: utf-8

# Q1.<br>
# 
# h(12): (3 x 12 + 5) mod 11 = 8<br>
# h(44): (3 x 44 + 5) mod 11 = 5<br> 
# h(13): (3 x 13 + 5) mod 11 = 0<br>
# h(88): (3 x 88 + 5) mod 11 = 5<br>
# h(23): (3 x 23 + 5) mod 11 = 8<br>
# h(94): (3 x 94 + 5) mod 11 = 1<br>
# h(11): (3 x 11 + 5) mod 11 = 5<br>
# h(39): (3 x 39 + 5) mod 11 = 1<br>
# h(20): (3 x 20 + 5) mod 11 = 10<br>
# h(16): (3 x 16 + 5) mod 11 = 9<br>
# h(5): (3 x 5 + 5) mod 11 = 9<br>

#     0	1	2	3	4	5	6	7	8	9	10
#     13	94				44			12	16	
#           39				88			23	5	
#                             11					
# 
# ![image.png](attachment:image.png)

# Q2.<br>
# 
#     0	1	2	3	4	5	6	7	8	9	10
#     13	94	39	16	5	44	88	11	12	23	20
# ![image.png](attachment:image.png)

# Q3.<br>
# h(12): 8<br>
# h(44): 5<br>
# h(13): 0<br>
# h(88): 5 + 1 = 6<br>
# h(23): 8 + 1 = 9<br>
# h(94): 1<br>
# h(11): 0 + 3 = 3<br>
# h(39): 1 + 1 + 2<br>
# h(20): 10<br>
# h(16): 4 + 3 = 7<br>
# h(5): cannot be loaded

#     0	1	2	3	4	5	6	7	8	9	10
#     13	94	39	11		44	88	16	12	23	20
# 
# ![image.png](attachment:image.png)

# Q4.<br>
# h(12): 8<br> 
# h(44): 5<br>
# h(13): 0<br>
# h(88): 5 + (7 - (88mod7)) (x 3) = 3<br>
# h(23): 8 + (7 - (23mod7)) = 2<br>
# h(94): 1 + (7 - (94mod7)) = 6<br>
# h(11): 5 + (7 - (11mod7)) = 9<br>
# h(39): 1<br>
# h(20): 10<br>
# h(16): 9 - (7 - (16mod7)) (x 4) = 7<br>
# h(5): 9 - (7 - (5mod7)) (x 3) = 4<br>

#     0	1	2	3	4	5	6	7	8	9	10
#     13	39	23	88	5	44	94	16	12	11	20
# 
# ![image.png](attachment:image.png)

# Q5.<br>
# 
#     0	1	2	3	4	5	6	7	8	9	10	11	12	13	14	15	16	17	18	19
#               12   18	41	  36	25		54			  38	10		  90	28			
# ![image.png](attachment:image.png)

# Q6.<br>
# 
# To determine the postition of the first zero in a row, a binary search must be performed on each row. Then all of the numbers must be added to obtain the number of 1's in A. Performing a binary search algorithm on each row of the matrix once will take O(n log n) time.
