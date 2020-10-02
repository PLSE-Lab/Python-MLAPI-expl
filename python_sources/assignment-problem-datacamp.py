#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import print_function
# import necessary dependencies pywrapgraph

import time

def main():
  cost = create_data_array()
  # define two variables named rows & cols equal to the size of the cost matrix
    
  assignment = pywrapgraph.LinearSumAssignment()
  # write a single nested for loop where index =  (worker & task) & length = (rows & cols) of parent & child arerespectively
  # you might get a hint from the below which are the statement of the for loop

      if cost[worker][task]:
        assignment.AddArcWithCost(worker, task, cost[worker][task])
  solve_status = assignment.Solve()
  if solve_status == assignment.OPTIMAL:
    # Print the optimal assignment
    print()
    for i in range(0, assignment.NumNodes()):
      print('Worker %d assigned to task %d.  Cost = %d' % (
            i,
            assignment.RightMate(i),
            assignment.AssignmentCost(i)))
  elif solve_status == assignment.INFEASIBLE:
    print('No assignment is possible.')
  elif solve_status == assignment.POSSIBLE_OVERFLOW:
    print('Some input costs are too large and may cause an integer overflow.')
def create_data_array():
  cost = [[90, 76, 75, 70],
          [last_2_digits_of_your_roll_number, 85, 55, 65],
          [125, 95, 90, 105],
          [45, 110, 95, 115]]
  return cost
if __name__ == "__main__":
  start_time = time.clock()
  main()
  print()
  print("Time =", time.clock() - start_time, "seconds")


# In[ ]:




