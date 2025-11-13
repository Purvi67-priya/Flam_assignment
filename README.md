# 📘 Parametric Curve Modeling and Optimization

## 🧮 Problem Statement

We are given the following **parametric equation** of a curve:

\[
\begin{aligned}
x(t) &= t\cos(\theta) - e^{M|t|}\sin(0.3t)\sin(\theta) + X \\
y(t) &= 42 + t\sin(\theta) + e^{M|t|}\sin(0.3t)\cos(\theta)
\end{aligned}
\]

The goal is to **find the unknown parameters** \( \theta, M, X \) that best fit a given set of data points \((x_i, y_i)\) for \(6 < t < 60\).

---

## ⚙️ Approach

### 1. Data Loading
The provided CSV file (`xy_data.csv`) contains \((x, y)\) coordinates sampled uniformly in the range \(6 \le t \le 60\).

### 2. Model Construction
The parametric model is implemented as:

\[
\begin{aligned}
x_{pred}(t) &= t\cos(\theta) - e^{M|t|}\sin(0.3t)\sin(\theta) + X \\
y_{pred}(t) &= 42 + t\sin(\theta) + e^{M|t|}\sin(0.3t)\cos(\theta)
\end{aligned}
\]

### 3. Objective Function
We minimize the **mean Euclidean distance** between observed and predicted points:

\[
J(\theta, M, X) = \frac{1}{N}\sum_{i=1}^{N} \sqrt{(x_i - x_{pred,i})^2 + (y_i - y_{pred,i})^2}
\]

### 4. Optimization Details
- **Algorithm:** L-BFGS-B (bounded optimization)
- **Parameter bounds:**
  | Parameter | Range |
  |------------|--------|
  | \(\theta\) | [0°, 50°] |
  | \(M\) | [-0.05, 0.05] |
  | \(X\) | [0, 100] |
- **Initial Guess:** [25, 0, 50]

---

## 🧩 Best-Fit Parameters

| Parameter | Value |
|------------|--------|
| \(\theta\) | **30.0441°** |
| \(M\) | **−0.00528** |
| \(X\) | **55.3473** |

---

## 🧠 Defining the Parameter tᵢ

Since the dataset only contains \(x_i\) and \(y_i\) coordinates, the parameter values \(t_i\) were **not given**.  
However, we know that the points lie on the curve for \(6 < t < 60\). Therefore, we **assumed uniform spacing** between these limits.

### 🔹 Derivation of the Formula

If we want to divide a range \([a, b]\) into \(N\) evenly spaced points, the step size is:

\[
\Delta t = \frac{b - a}{N - 1}
\]

Hence each point is computed as:

\[
t_i = a + (i - 1)\Delta t
\]

Substituting \(a = 6\) and \(b = 60\):

\[
\boxed{t_i = 6 + \frac{(i - 1)(60 - 6)}{N - 1}}
\]

This ensures:
- \(t_1 = 6\)
- \(t_N = 60\)
- All other \(t_i\) values are equally spaced in between.

In Python, this is achieved by:
```python
t = np.linspace(6, 60, N)
```

---

## 🧮 Python Implementation

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load CSV
data = pd.read_csv("xy_data.csv")
x_obs, y_obs = data.iloc[:, 0].values, data.iloc[:, 1].values
t = np.linspace(6, 60, len(x_obs))

# Parametric model
def model(params, t):
    theta_deg, M, X = params
    theta = np.deg2rad(theta_deg)
    exp_term = np.exp(M * np.abs(t))
    x_pred = t*np.cos(theta) - exp_term*np.sin(0.3*t)*np.sin(theta) + X
    y_pred = 42 + t*np.sin(theta) + exp_term*np.sin(0.3*t)*np.cos(theta)
    return x_pred, y_pred

# Objective function
def objective(params):
    x_pred, y_pred = model(params, t)
    return np.mean(np.sqrt((x_obs - x_pred)**2 + (y_obs - y_pred)**2))

# Optimize
bounds = [(0, 50), (-0.05, 0.05), (0, 100)]
res = minimize(objective, [25, 0, 50], bounds=bounds, method='L-BFGS-B')
theta, M, X = res.x

print(f"Theta = {theta:.6f}°, M = {M:.6f}, X = {X:.6f}")

# Plot
x_fit, y_fit = model(res.x, t)
plt.scatter(x_obs, y_obs, s=10, alpha=0.6, label='Observed points')
plt.plot(x_fit, y_fit, 'r', lw=2, label='Fitted curve')
plt.xlabel('x'); plt.ylabel('y'); plt.legend(); plt.grid(True)
plt.title('Observed Data vs Fitted Parametric Curve')
plt.show()
```

---

## 📈 Results and Visualization

The fitted curve (red) closely matches the dataset points (blue/orange), confirming the correctness of the model and the accuracy of parameter estimation.

---

## 📘 Step-by-Step Summary of the Process

1. **Understand the problem** — find parameters \(\theta, M, X\) for the curve using given data points.
2. **Load the CSV** — read \(x_i, y_i\) values from the provided file.
3. **Assign t-values** — use uniform spacing between 6 and 60:
   \(t_i = 6 + (i - 1)(60 - 6)/(N - 1)\)
4. **Build the model** — code the parametric equations.
5. **Define the loss function** — mean Euclidean distance between observed and predicted points.
6. **Optimize parameters** — use `scipy.optimize.minimize` (L-BFGS-B) with bounds.
7. **Plot results** — visualize observed points vs fitted curve.
8. **Generate final expressions** — suitable for Desmos and reports.
9. **Document** — record approach, results, and explanation in this README.

---

#✅ Conclusion

-The optimized model provides an excellent fit to the observed dataset.

-The exponential term e^(M|t|) introduces a slight damping effect, reducing amplitude for larger |t| values since M is negative.

-The optimized parameters — θ = 30.04°, M = -0.00528, and X = 55.35 — accurately represent the best-fit curve.

-This workflow demonstrates a complete process for parametric curve fitting, optimization, and model validation using Python.
---
