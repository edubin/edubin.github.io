## Introduction to Locally Weighted Regression ##

**Main Idea:** In Locally Weighted Regression (LWR), a regression is performed around a single point of interest. LWR fits a nonparametric regression curve to a scatterplot.

While trends and associations in regression are generally nonlinear; these trends can be interpreted linearly by using LWR. Training data is weighted by each data point's distance from the point of interest using a kernel (more on this later). Then, a regression is computed using the *weighted* points. 

The independent observations are the rows of a matrix *X*. Features are represented in columns, denoted by *p*. As such, every row is a vector in $$\mathbb{R}^p$$. We use a method called *metric* to compute these distances between each observation and the point of interest. Since observations contain multiple features, we interpret them at them as vectors in a finite-dimensional Euclidean space. The vector equation is:

$$ dist(\vec{v},\vec{w})=\sqrt{(v_1-w_1)^2+(v_2-w_2)^2+...(v_p-w_p)^2}$$

There are *n* different weight vectors because there are $n$ observations.

The predictions we make are a linear combination of the actual observed values of the dependent variable. This is expressed in the equation:

$$\large \hat{y} = X(X^TWX)^{-1}(X^TWy)$$

For locally weighted regression, $$\hat{y}$$ is obtained as different linear combinationz of the values of y.

# Visual Intuition for Locally Weighted Regression

![\label{fig:locregression}](/project1/ps3-660x280.png)

This image illistrates how the locally weighted regression is optimized with tau, or bandwidth. When tau is too large or too small, the regression is not fitting the data appropriately. Data validation and scatterplots can help us see if the value tau is not optinal (1). A larger tau will result in a smoother curve, however this is not always desired. To address this, different values for tau, along with different kernels which alter the shape of the curve.

# Kernal Selection

There are a wide array of kernels that support locally weighted regression. When selecting a kernel, the goal is to find one in which the function has one local maximum that has a compact support. Some kernels we can use are:

1.   The Exponential Kernel (based on exponential function)

$$ K(x):= e^{-\frac{\|x\|^2}{2}}$$


2.   The Tricubic Kernel

$$ K(x):=\begin{cases}
  (1-\|x\|^3)^3&if&\|x\|<1 \\
0 & \text{otherwise}
\end{cases}
$$

```python
def tricubic(x):
  return np.where(np.abs(x)>1,0,(1-np.abs(x)**3)**3)   
 ```

3.   The Epanechnikov Kernel

$$ K(x):=\begin{cases}
\frac{3}{4}(1-\|x\|^2)&if&\|x\|<1 \\
0 & \text{otherwise}
\end{cases}
$$

```python
# Epanechnikov Kernel
def Epanechnikov(x):
  return np.where(np.abs(x)>1,0,3/4*(1-np.abs(x)**2)) 
```

3.   The Quartic Kernel

$$ K(x):=\begin{cases}
\frac{15}{16}(1-\|x\|^2)^2 &if&\|x\|<1 \\
0 &\text{otherwise}
\end{cases}
$$

```python
# Quartic Kernel
def Quartic(x):
  return np.where(np.abs(x)>1,0,15/16*(1-np.abs(x)**2)**2)
```

It is important that the kernel function is vectorized before it is used for LWR. The functions in numpy are already vectorized, so we can use their functions to avoid an extra step of vectorizing the function.

When defining the kernel function, tau represents how wide the kernel is, and x represents the center (or point of interest).

```python
def kernel_function(xi,x0,kern, tau): 
    return kern((xi - x0)/(2*tau))
```

When the vector of data is large, we can apply the kernel around each point xi to get values that correspond to each value of x.

```python
def weights_matrix(x,kern,tau):
  n = len(x)
  return np.array([kernel_function(x,x[i],kern,tau) for i in range(n)]) 
```

## Lowess Function

The Lowess Function is used to perform Locally Weighted Regression. Parameters x and y are arrays that contain an equal number of element. Each pair x[i] and y[i] define a data point in the scatterplot. The function returns the estimated values of y. 

There are a few ways to approach the Lowess Function. The first way is by solving a system of linear equations with fixed data and weights:

``` python
def lowess(x, y, kern, tau):
    n = len(x)
    yest = np.zeros(n)

    #Initializing all weights from the bell shape kernel function    
    w = np.array([kern((x - x[i])/(2*tau)) for i in range(n)])     
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
        A = np.array([[np.sum(weights), np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights * x * x)]])
        beta = linalg.solve(A, b) # A*beta = b
        # beta, res, rnk, s = linalg.lstsq(A, b)
        yest[i] = beta[0] + beta[1] * x[i] 

    return yest
  ```
The second way is by solving a linear equation for each weight:

``` python
def lowess(x, y, kern, tau=0.05):
    n = len(x)
    yest = np.zeros(n)
    
    #Initializing all weights from the bell shape kernel function    
    w = weights_matrix(x,kern,tau)    
    
    #Looping through all x-points
    for i in range(n):
        weights = w[:, i]
        lm.fit(np.diag(w[:,i]).dot(x.reshape(-1,1)),np.diag(w[:,i]).dot(y.reshape(-1,1)))
        yest[i] = lm.predict(x[i].reshape(-1,1)) 

    return yest
```

Finally, there is the approach by Alex Gramfort, which incorperates the hyperparameter f. F represents a smootheing span (this is what the kernel does in the previous two functions).

``` python
def lowess_ag(x, y, f, iter):
    n = len(x)
    r = int(ceil(f * n))
    h = [np.sort(np.abs(x - x[i]))[r] for i in range(n)]
    w = np.clip(np.abs((x[:, None] - x[None, :]) / h), 0.0, 1.0)
    w = (1 - w ** 3) ** 3
    yest = np.zeros(n)
    delta = np.ones(n)
    for iteration in range(iter):
        for i in range(n):
            weights = delta * w[:, i]
            b = np.array([np.sum(weights * y), np.sum(weights * y * x)])
            A = np.array([[np.sum(weights), np.sum(weights * x)],
                          [np.sum(weights * x), np.sum(weights * x * x)]])
            beta = linalg.solve(A, b)
            yest[i] = beta[0] + beta[1] * x[i]

        residuals = y - yest
        s = np.median(np.abs(residuals))
        delta = np.clip(residuals / (6.0 * s), -1, 1)
        delta = (1 - delta ** 2) ** 2

    return yest
 ```

Ultimatly, different kernels and different taus will help us to generate better estimations.

## Validation
To validate our function, we can use a Train-Test split. To do this, we can either generate data, or import data to separate into testing and training data. We then can plot the kernal regression values of y with the true data in order to gauge the effectiveness of the regression.

Here is an example of some code to generate a scatterpot to validate data, along with the resulting plot:
``` python
plt.figure(figsize=(10,6))
plt.scatter(x,ynoisy,ec='blue',alpha=0.5)
plt.plot(x,yest,color='red',lw=2,label='Kernel Regression')
plt.plot(x,y,color='green',lw=2,label='Truth')
plt.legend()
plt.show
```

![\label{fig:exampleplot}](/project1/exampleplot.png)

## References

1. [LWR Visualization](https://www.geeksforgeeks.org/locally-weighted-linear-regression-using-python/)
