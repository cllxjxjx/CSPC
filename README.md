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



## PW3 - Session 1: Characterise the Data

**Dataset Summary:**
- Loaded `heart.csv` (~300 patients) isolating continuous features: `age`, `chol`, `trestbps`, `thalach`.

**Shape & Distribution Descriptions:**
- **age**: Roughly symmetric, bell-shaped distribution centered around 54 years.
- **chol** (cholesterol): Slightly right-skewed with a few high-value outliers above 400 mg/dL.
- **trestbps** (resting blood pressure): Moderate right skew with slight clustering around typical clinical intervals (e.g., 120, 130, 140 mmHg).
- **thalach** (max heart rate): Moderately left-skewed, peaking around 150–160 bpm.

**Normality Verdicts:**
- **age**: Approximately normal based on graphical symmetry and mild skewness.
- **chol**: Not normal due to right-tail skewness and extreme outliers (Shapiro-Wilk p < 0.05).
- **trestbps**: Not normal due to right skew and thresholding effects (Shapiro-Wilk p < 0.05).
- **thalach**: Not normal due to left skewness (Shapiro-Wilk p < 0.05).

## PW3 - Session 2: Draw Conclusions

**Is the effect real? (Heart Disease vs Thalach):**
- A Mann-Whitney U test confirmed that heart disease patients reach a significantly lower maximum heart rate (`thalach`) compared to healthy patients ($p < 0.001$).
- Displaying group means with Standard Error of the Mean (SEM) error bars clearly separates the two populations beyond visual uncertainty.

**Do two things move together? (Age vs Thalach):**
- Age and `thalach` display a moderate negative correlation ($r \approx -0.40, p < 0.001$), demonstrating that maximum achievable heart rate decreases systematically as age increases.

**Chemical-Exposure Mystery (Confounding Variables):**
- **Naive Analysis:** At face value, one chemical appeared far more strongly associated with tumor malignancy score.
- **Controlling for Pollution Index:** Restricting the cohort to patients with similar area pollution indices ($40 < \text{pollution\_index} < 60$) caused the naive association to collapse. The third variable (`pollution_index`) was the true underlying driver influencing both chemical exposure and malignancy score.

**Bonus (Category Unpredictability):**
- Calculated Shannon entropy on the target class proportions yields a value close to $1.0\text{ bit}$, indicating a nearly balanced distribution that represents maximum unpredictability for binary classification.

## PW3 --- Data Characterisation and Chemical Mystery Report

### Session 1: Data Characterisation & Normality Verdicts
* **age:** Approximately normal (bell-shaped distribution, low skewness).
* **chol:** Not normal (right-skewed with extreme high-value outliers above 400 mg/dL, Shapiro-Wilk test $p < 0.05$).
* **trestbps:** Not normal (right-skewed due to clinical measurement clustering, Shapiro-Wilk test $p < 0.05$).
* **thalach:** Not normal (left-skewed toward higher max heart rates, Shapiro-Wilk test $p < 0.05$).

### Session 2: Heart Rate & Age Analysis
* **Heart Rate Comparison (`thalach`):** Non-parametric Mann-Whitney U test confirmed that heart disease patients reach a significantly lower maximum heart rate ($\approx 139$ bpm) compared to healthy patients ($\approx 158$ bpm, $p < 0.001$).
* **Age vs. Heart Rate Relationship:** Spearman correlation ($r \approx -0.40, p < 0.001$) demonstrated a significant negative trend: maximum heart rate decreases systematically as age increases.

### Chemical Exposure Mystery: Correlation vs. Causation
* **Naive Conclusion:** At face value, `benzene` showed a strong positive correlation with tumor malignancy, making it look guilty, while `cadmium` appeared weak.
* **Controlling for Confounding Variable (`pollution_index`):** When restricting the cohort to patients in the same pollution band ($40 < \text{pollution\_index} < 60$), the correlation between `benzene` and malignancy completely vanished ($r \approx 0$). Meanwhile, `cadmium` showed a strong positive association with malignancy across all pollution tiers.
* **Final Verdict:** `cadmium` is the actual chemical that drives malignancy. `benzene` only appeared guilty because high benzene concentrations co-occurred with high area pollution levels, which acted as a confounding variable.
