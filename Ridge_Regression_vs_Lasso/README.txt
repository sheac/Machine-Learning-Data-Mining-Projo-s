The script in ridge_vs_lasso.py calls functions in util_functions.py to compare the performance of Lasso and Ridge Regression.

Specifically it:
1) Plots the raw data as a scatterplot
2) Graphs the regularization paths against different degrees of freedom for the Ridge method
3) Uses 10-fold Cross Validation to graph the squared errors against different degrees of freedom for the Ridge method
4) Does the same for Lasso method, using the Shooting algorithm

I guess it may be redundant to say, but to run this puppy, just run the driver class ridge_vs_lasso.py.
It may not be redundant to mention that the functions found in util_functions.py are probably useful outside of this specific context. 
Feel free to use them, but please cite me (Shealen Clare) and the class that asked me to figure out and then construct this stuff for homework: (UBC Computer Science 340: Machine Learning).

Thanks :)
