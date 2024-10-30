from numpy import linalg as LA

def Newton(f, x, dfdx, epsilon=1.0E-7, N=100, store=False):
    """
    Newton's method for finding roots of a function.

    Parameters:
    f (function): The function to find roots for.
    x (float or array): The initial guess.
    dfdx (function): The derivative of the function.
    epsilon (float, optional): The tolerance for convergence. Defaults to 1.0E-7.
    N (int, optional): The maximum number of iterations. Defaults to 100.
    store (bool, optional): Whether to store the iteration history. Defaults to False.

    Returns:
    x (float or array): The root of the function.
    n (int): The number of iterations.
    info (list, optional): The iteration history if store is True.
    """
    
    # Initialize variables
    f_value = f(x)
    dfdx_value = dfdx(x)
    n = 0
    if store: info = []
    
    while LA.norm(f_value) > epsilon and n <= N:
        
        # Check for singular derivative
        if LA.norm(dfdx_value) < 1E-14:
            raise ValueError("Newton: f'(%g)=%g" % (x, LA.norm(dfdx_value)))

        if f_value.size > 1:
            x = x - LA.solve(dfdx_value, f_value)
        else:
            x = x - f_value / dfdx_value
        
        # Update variables
        f_value = f(x)
        dfdx_value = dfdx(x)
        n += 1
        if store:
            info.append(x)
    
    if store:
        return x, n, info
    else:
        return x, n, f_value
    
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

    # Initialize variables
    R = dgdx(x[1]) @ dgdx(x[0]).T
    Delta_Lambda = g(x[1]) / R if R.ndim <= 1 else LA.solve(R, g(x[1]))
    m = 0

    if store:
        info = [(m, Delta_Lambda, x[1])]

    # Fixed point iteration
    while LA.norm(g(x[1])) > tol and m < M:
        # Update x
        if R.ndim <= 1:
            x[1] -= dgdx(x[0]) * Delta_Lambda
        else:
            x[1] -= dgdx(x[0]).T @ Delta_Lambda

        # Update R and Delta_Lambda
        R = dgdx(x[1]) @ dgdx(x[0]).T
        Delta_Lambda = g(x[1]) / R if R.ndim <= 1 else LA.solve(R, g(x[1]))

        # Update m and x[0]
        m += 1
        x[0] = x[1]

        # Store iteration history
        if store:
            info.append((m, Delta_Lambda, x[1]))
        
    # Check convergence
    assert m < M, "Nonlinear solver did not converge"

#%% Testing
from numpy import sin, cos, exp, linspace, pi, array
import matplotlib.pyplot as plt
    
def _g(x):
    return exp(-0.1*x**2)*sin(pi/2*x)

def _dg(x):
    return -2*0.1*x*exp(-0.1*x**2)*sin(pi/2*x) + \
           pi/2*exp(-0.1*x**2)*cos(pi/2*x)

def _test():

    #x0 = float(sys.argv[1])
    x0 = 0.1
    x, info = Newton(_g, x0, _dg, store=True)
    print('root: %.16g' % x)
    for i in range(len(info)):
        print('Iteration %2d: f(%g)=%g, dF(%g) =%g' % \
              (i, info[i][0], info[i][1], info[i][0], info[i][2]))

    x = array(info[:][0])
    y = _g(x)
    plt.plot(x, y, 'r.', linspace(-7, 7, 100), _g(linspace(-7, 7, 100)), 'b.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Newton iterates')
    plt.savefig('tmp.pdf')

if __name__ == '__main__':
    _test()
    