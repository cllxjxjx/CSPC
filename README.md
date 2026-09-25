# Lab A Project
What I built:
I built CSPC course repository structure,conda envirionment,.gitignore,branches,tests and performance comparison script.

Speed comparison:
loop:2.7099 seconds
numpy:0.0003 seconds
NumPy is 8587.11x faster than pure python loop.

Tests:
All the tests passed succesfully
 
 Conclusion:
 In this pw,I learned how to set up a envioriment with conda,using Git and GitHub.Additionally,i observed significant performance increase
 using NumPy operations compared to pure-Python loops for simulating atomic decay.

## PW1 Lab B

I observed that the `decay_observed.csv` dataset contains time and count measurements showing a rapid decrease over time.
The observed scatter data matches the analytical decay law very well.It follows the exact same exponential trajectory.
The Snakemake tracks file timestamps to automatically regenerate `figure.png` whenever the dataset or plotting script is modified.


##PW2 Lab A
In this experiment i got the mean of acceleration -8.75 m/s*s which is noisy.The acceleration is extremely noisy because numerical differentiation compares nearby data points and amplifies measurement noise, and taking the derivative twice swamped the real signal.
We used integrals for position and regardless of the acceleration being noisy since integral consists of summing the summing process mostly cancels the  noise so we got 0.64m.Derivative increase the noise but integral cancels them.

Conclusion:
In this pw,i learned about relations between velocity,acceleration and position.I also learned how integrals and derivatives affect noisiness in measurements.
