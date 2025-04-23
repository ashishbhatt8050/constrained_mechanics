from numpy import linalg as LA

def Newton(f, x, dfdx, tol, M, store):
    """
    Newton's method for finding roots of a function.

    Parameters:
    f (function): The function to find roots for.
    x (float or array): The initial guess.
    dfdx (function): The derivative of the function.
    tol (float): The tolerance for convergence. Defaults to 1.0E-7.
    M (int): The maximum number of iterations. Defaults to 100.
    store (bool): Whether to store the iteration history. Defaults to False.

    Returns:
    x (float or array): The root of the function.
    n (int): The number of iterations.
    info (list, optional): The iteration history if store is True.
    """
    
    # Initialize variables
    f_value = f(x)
    dfdx_value = dfdx(x)
    m = 0
    if store: info = []
    
    while LA.norm(f_value) > tol and m < M:
        
        # Check for singular derivative
        if LA.norm(dfdx_value) < 1E-14:
            raise ValueError("Newton: f'(%g)=%g" % (x, LA.norm(dfdx_value)))
        
        try:
            x = x - LA.solve(dfdx_value, f_value)
        except LA.LinAlgError:
            x = x - f_value / dfdx_value
        
        # Update variables
        f_value = f(x)
        dfdx_value = dfdx(x)
        m += 1
        if store:
            info.append(x)
        
    # Check convergence
    if m >= M:
        raise RuntimeError("Newton: nonlinear solver did not converge")
    
    if store:
        return x, m, info
    else:
        return x, m, f_value
    
def fixed_point(g, x, dgdx, tol, M, store):
    """
    Fixed point iteration method.

    Parameters:
    g (function): The function to find the fixed point for.
    x (array): The initial guess.
    dgdx (function): The derivative of the function.
    tol (float): The tolerance for convergence.
    M (int): The maximum number of iterations.
    store (bool): Whether to store the iteration history.

    Returns:
    x (array): The fixed point.
    info (list, optional): The iteration history if store is True.
    """

    # Initialize counter
    m = 0
    
    if isinstance(dgdx, tuple):
        dgdx_0, dgdx_1 = dgdx[0], lambda x: dgdx[1](0.5 * (x[0]+x[1]))
    elif isinstance(dgdx, list):
        dgdx_0, dgdx_1 = dgdx[0], lambda x: dgdx[1](x)
    else:
        raise TypeError("dgdx must be a tuple or list")

    if store:
        info = [(m, x[1])]

    # Fixed point iteration
    while LA.norm(g(x[1])) > tol and m < M:
        try:
            # Update R and Delta_Lambda
            R = dgdx_0(x[1]) @ dgdx_1(x).T
            Delta_Lambda = LA.solve(R, g(x[1]))

            x[1] -= dgdx_1(x).T @ Delta_Lambda
        except (TypeError, AttributeError):
            # Handle the case when g is a scalar
            R = dgdx_0(x[1]) * dgdx_1(x)
            Delta_Lambda = g(x[1]) / R
            x[1] -= dgdx_1(x) * Delta_Lambda
            
        # Update m and x[0]
        m += 1
        # x[0] = x[1]

        # Store iteration history
        if store:
            info.append((m, x[1], Delta_Lambda))
        
    # Check convergence
    if m >= M:
        raise RuntimeError("Nonlinear solver did not converge")

#%% Testing
if __name__ == "__main__":
    
    # Test Newton's method
    f = lambda x: x**2 - 2
    dfdx = lambda x: 2*x
    x, n, info = Newton(f, 1, dfdx, store=True)
    print("Root of x^2 - 2: ", x)
    print("Number of iterations: ", n)
    print("Iteration history: ", info)
    
    # Test fixed point iteration
    g = lambda x: x**2 - 2
    dgdx = lambda x: 2*x
    x = [1, 1]
    fixed_point(g, x, dgdx, 1.0E-7, 100, store=True)
    print("Fixed point of x^2 - 2: ", x)
