CS489 A1 - documentation
xiling wu
20561976
x242wu
==========================================================================================================================================================================================================
Python version : 3.6

Python package imported:

import csv   # for csv read, but I found it useless since numpy already able to read it.
from numpy import *
import numpy as np  # all calculation rely on numpy
import matplotlib.pyplot as plt  # draw the plot
import timeit       # used for estimate the time complexity on E2Q5

==========================================================================================================================================================================================================
script name and corresponding questions:
(all test data csv has been hard coded for convenience,however A1E1Q1.py have prompt to ask for max pass, and in the rest script, max pass are set on line 7, which could be changed eaisly)

Exercise 1:
    question 1: A1E1Q1.py (data set has been hard coded and script will ask for max pass, the attached png was generated when max pass is 500)
    question 2: A1E1Q2.py
    question 3: N/A
    question 4: A1E1Q4.py
    question 5: A1E1Q5.py
    question 6: For the perceptron: A1E1Q6-perceptron.py
                For the winnow: A1E1Q6-winnow.py
                For the step size tune: A1E1Q6-winnow-tunestep.py (it takes around 30 mins to run because I set up step range from 0 - 12, the range can be modified on line 67 for smaller test range. And I will attach the result at the bottom of this file)

Exercise 2:
    question 1: A1E2Q1.py
    question 2: A1E2Q2.py
    question 3: A1E2Q3.py
    question 4: A1E2Q4.py
    question 5: A1E2Q5.py(may take 10 - 15 mins to run)
    optional for Q5:  the 100 random features test mentioned is implemented in file A1E2Q5-100featuretest.py

==========================================================================================================================================================================================================

A1E1Q6 step size tune result:(running A1E1Q6-winnow-tunestep.py)
[step size, lowest mistake get within step size 500](asceding order)
[(5.4000000000000004, 1792), (5.6000000000000005, 1833), (4.5, 1837), (4.7000000000000002, 1837), (5.0, 1839), (5.8000000000000007, 1839), (4.4000000000000004, 1840), (5.2000000000000002, 1841), (6.1000000000000005, 1841), (3.7000000000000002, 1842), (8.3000000000000007, 1842), (4.9000000000000004, 1846), (4.8000000000000007, 1847), (5.5, 1847), (4.2000000000000002, 1848), (6.2000000000000002, 1848), (3.5, 1849), (6.7000000000000002, 1849), (3.9000000000000004, 1850), (6.6000000000000005, 1850), (8.4000000000000004, 1850), (4.6000000000000005, 1853), (4.1000000000000005, 1854), (3.4000000000000004, 1855), (7.0, 1855), (8.2000000000000011, 1855), (5.9000000000000004, 1857), (3.1000000000000001, 1858), (4.2999999999999998, 1858), (6.0, 1858), (5.7000000000000002, 1860), (5.1000000000000005, 1861), (5.3000000000000007, 1861), (4.0, 1862), (7.7000000000000002, 1862), (7.3000000000000007, 1863), (7.6000000000000005, 1863), (3.8000000000000003, 1866), (8.0999999999999996, 1866), (6.5, 1867), (6.4000000000000004, 1869), (7.1000000000000005, 1869), (3.6000000000000001, 1870), (7.9000000000000004, 1871), (6.8000000000000007, 1875), (3.2000000000000002, 1876), (8.0, 1876), (7.4000000000000004, 1879), (7.5, 1880), (7.8000000000000007, 1880), (6.3000000000000007, 1885), (7.2000000000000002, 1885), (6.9000000000000004, 1886), (2.9000000000000004, 1892), (3.3000000000000003, 1896), (8.5, 1902), (8.5999999999999996, 1910), (3.0, 1911), (2.7000000000000002, 1913), (2.8000000000000003, 1915), (2.6000000000000001, 1917), (2.2000000000000002, 1926), (2.3000000000000003, 1926), (2.5, 1944), (2.0, 1946), (2.4000000000000004, 1947), (2.1000000000000001, 1949), (8.7000000000000011, 1956), (8.9000000000000004, 1957), (8.8000000000000007, 1960), (9.0, 1969), (1.9000000000000001, 1970), (1.8, 1971), (1.6000000000000001, 1972), (1.5, 1986), (1.7000000000000002, 1989), (9.0999999999999996, 1994), (1.4000000000000001, 1995), (1.3, 2017), (1.2000000000000002, 2037), (1.1000000000000001, 2048), (9.2000000000000011, 2072), (1.0, 2077), (0.90000000000000002, 2094), (0.80000000000000004, 2123), (0.70000000000000007, 2163), (9.3000000000000007, 2175), (0.60000000000000009, 2193), (0.5, 2244), (0.10000000000000001, 2301), (0.40000000000000002, 2307), (0.30000000000000004, 2376), (0.0, 2385), (0.20000000000000001, 2466), (9.4000000000000004, 2837), (9.5, 2837), (9.6000000000000014, 2837), (9.7000000000000011, 2837), (9.8000000000000007, 2837), (9.9000000000000004, 2837), (10.0, 2838), (10.100000000000001, 2838), (10.200000000000001, 2838), (10.300000000000001, 2873), (10.4, 2881), (10.5, 2883), (10.600000000000001, 2883), (10.700000000000001, 2883), (10.800000000000001, 2883), (10.9, 2892), (11.0, 2897), (11.100000000000001, 2897), (11.200000000000001, 2993), (11.300000000000001, 2993), (11.4, 2993), (11.5, 2993), (11.600000000000001, 3008), (11.700000000000001, 3010), (11.800000000000001, 3010), (11.9, 3010)]
So, step size as 5.4 is used to generate the plot in the pdf.
If you would like to run this script, you may want to limit the tune range to (5,6,0.1) to get a shorter run time.