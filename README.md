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


## PW2 Lab A
In this experiment i got the mean of acceleration -8.75 m/s*s which is noisy.The acceleration is extremely noisy because numerical differentiation compares nearby data points and amplifies measurement noise, and taking the derivative twice swamped the real signal.
We used integrals for position and regardless of the acceleration being noisy since integral consists of summing the summing process mostly cancels the  noise so we got 0.64m.Derivative increase the noise but integral cancels them.I also used the formula sqrt(vx**2 + vy**2) as numpy and made the graph of it.

Conclusion:
In this pw,i learned about relations between velocity,acceleration and position.I also learned how integrals and derivatives affect noisiness in measurements.

## PW2 Lab B
For f(x) funct:
Gradient Descent: 2.9999963220107015
Newton Method: 3.0
SLSQP method:3.0
For g(x) funct x = 0:
Gradient Descent: -1.3008397603384068
Newton Method: 0.16993844331159128 Second derivative:-5.653451105817997
Newton shows Max
SLSQP method:-1.3008569447162184
When x=2:
Gradient Descent: -1.3008397603384068
Newton Method: 1.1309011226299859 Second derivative:9.347248189989148
Newton shows Minimum
SLSQP method:-1.3006394477423422
In this exp in the i applied gradient descent,newton and SLSQP to the functions f(x) and g(x).In function g(x) when x0 = 0 with newton method it stops at 0.17  but since its second derivative is -5.65 (<0) it is not minimum point its maximum stasionary point.But when x=2 the second derivative is postive so it shows local minimum.
When the second derivative is minimum newton method doesnt agree with others but when it is >0 they agree with each other.

In part 3 of this pw rate constant k is 0.2618.
The errors in 'kinetics.cv' optimized as quadratic errors sum and we got k constant suitable for exp line with the help of SLSQP minimization.And the new graph made.

In Part 4 of this pw:
Balance:
n(H2) = 0.34 mol
n(I2) = 0.34 mol
n(HI) = 1.33 mol
Finding the roots and minimizing error's square determined the chemical equilibrium point with same accuracy.


