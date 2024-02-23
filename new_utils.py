"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""


def scale_data(X):
    
    scaled_X = (X - X.min()) / (X.max() - X.min())
    
    return scaled_X
