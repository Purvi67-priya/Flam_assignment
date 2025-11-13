📘 Parametric Curve Modeling and Optimization
This project focuses on estimating the unknown parameters
θ,M,and X of the following parametric curve:
x(t)y(t)​=tcos(θ)−e^M∣t∣sin(0.3t)sin(θ)+X
y(t)=42+tsin(θ)+eM∣t∣sin(0.3t)cos(θ)​
Given a dataset of observed points (𝑥𝑖,𝑦𝑖) the objective is to determine the best-fitting parameters that generate this curve.

🚀 1. Problem Overview

We are provided only with coordinate pairs (𝑥𝑖,𝑦𝑖).
The model parameter 𝑡 does not come with the dataset.
However, the curve is known to lie in the range:
                   6≤t≤60
To reconstruct the model accurately, we must estimate:
       -the hidden 𝑡𝑖 values,
			 -the parameters 𝜃,𝑀,𝑋
       -and obtain a smooth fitted curve.
