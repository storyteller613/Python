# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 14:46:49 2017

@author: JG
"""
#############################
#Importing entire text files
############################

# Open a file: file
file = open('moby_dick.txt', 'r')

# Print it
print(file.read())

# Check whether file is closed
print(file.closed)

# Close file
file.close()

# Check whether file is closed
print(file.closed)

##################################
#Importing text files line by line
##################################

# Read & print the first 3 lines
with open('moby_dick.txt') as file:
    print(file.readline())
    print(file.readline())
    print(file.readline())
    
with open('huck_finn.txt') as file:
    
with open('huck_finn.txt') as file:
    print(file.read())
    
