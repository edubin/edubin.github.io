

## Introduction to Locally Weighted Regression ##

**Main Idea:** Trends and associations are generally nonlinear; however, *locally*, trends can be interpreted linearly.

In this context, local properties are relative to a metric. A metric is a method by which we compute the distance between two observations. Observations contain multiple features, and if they are numeric, we can see them as vectors in a finite-dimensional Euclidean space.

The independent observations are the rows of the matrix $X$. Each row has a number of columns (this is the number of features) and we can denote it by $p.$ As such, every row is a vector in $\mathbb{R}^p.$ The distance between two independent observations is the Euclidean distance between the two represented $p-$dimensional vectors. The equation is:

$$ dist(\vec{v},\vec{w})=\sqrt{(v_1-w_1)^2+(v_2-w_2)^2+...(v_p-w_p)^2}$$

We shall have $n$ different weight vectors because we have $n$ different observations.

# Visual Intuition for Locally Weighted Regression

![\label{fig:locregression}](/project1/ps3-660x280.png)

This image illistrates how the locally weighted regression is optimized with tau, or bandwith. When tau is too large or too small, the regression is voerfitting or underfitting the training data. (1)

# Kernal Selection

There are many choices of kernels for locally weighted regression. The idea is to have a function with one local maximum that has a compact support.
- these make more sense when they are plotted

1.   The Exponential Kernel (based on exponential function)

$$ K(x):= e^{-\frac{\|x\|^2}{2}}$$


2.   The Tricubic Kernel

$$ K(x):=\begin{cases}
  (1-\|x\|^3)^3&if&\|x\|<1 \\
0 & \text{otherwise}
\end{cases}
$$

3.   The Epanechnikov Kernel (name of the person who found it, proved to be very efficient)

$$ K(x):=\begin{cases}
\frac{3}{4}(1-\|x\|^2)&if&\|x\|<1 \\
0 & \text{otherwise}
\end{cases}
$$

3.   The Quartic Kernel

$$ K(x):=\begin{cases}
\frac{15}{16}(1-\|x\|^2)^2 &if&\|x\|<1 \\
0 &\text{otherwise}
\end{cases}
$$

```python
#!/usr/local/bin/python3
# testargs.py

import sys

print ("{} is the name of the script." . format(sys.argv[0]))
print ("There are {} arguments: {}" . format(len(sys.argv), str(sys.argv)))

for ind, arg in enumerate(sys.argv):
    print ("[{}]: {} {}".format(ind,arg,sys.argv[ind]))
```

## Lowess Function

The lowess function fits a nonparametric regression curve to a scatterplot.
The arrays x and y contain an equal number of elements; each pair
(x[i], y[i]) defines a data point in the scatterplot. The function returns
the estimated (smooth) values of y.
The smoothing span is given by f. A larger value for f will result in a
smoother curve. The number of robustifying iterations is given by iter. The
function will run faster with a smaller number of iterations.

## References

1. [editior on Github](https://www.geeksforgeeks.org/locally-weighted-linear-regression-using-python/)

