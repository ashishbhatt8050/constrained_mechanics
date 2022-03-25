import numpy as np
import scipy as sp
from pylab import sum
from scipy.sparse import issparse
from scipy.sparse.linalg import spsolve, ArpackNoConvergence, ArpackError

def Newton(f, x, dfdx, epsilon=1.0E-7, N=100, store=False):
    f_value = f(x)
    n = 0
    if store: info = [(x, f_value, np.linalg.norm(dfdx(x)))]
    while np.linalg.norm(f_value) > epsilon and n <= N:
        dfdx_value = dfdx(x)
        if np.linalg.norm(dfdx_value) < 1E-14:
            raise ValueError("Newton: f'(%g)=%g" % (x, np.linalg.norm(dfdx_value)))

        if not issparse(dfdx_value):
            try:
                x = x - np.linalg.solve(dfdx_value, f_value)
            except np.linalg.LinAlgError:
                x = x - f_value/dfdx_value
            except:
                raise np.linalg.LinAlgError("Unable to solve the system")
        else:
            try:
                x = x - sp.sparse.linalg.spsolve(dfdx_value, f_value)
            except sp.linalg.LinAlgError:
                x = x - f_value/dfdx_value
            except:
                raise ArpackError("Unable to solve the sparse system")
            
        n += 1
        f_value = f(x)
        if store: info.append((x, f_value, dfdx_value))
    if store:
        return x, info
    else:
        return x, n, f_value
    
def fixed_point(g, x, dgdx, tol, M, store):
    # TODO: convert * to matrix multiplication
    # update x[0] below
    # try to avoid .T
    # m, Delta_Lambda = 0, g(x[1:2])/sum(dgdx(x[1:2])*dgdx(x[0:1]), axis=1)
    m, Delta_Lambda = 0, g(x[1:2])/((dgdx(x[1:2]).dot(dgdx(x[0:1]).T)).diagonal())
    
    if store: info = [(m, Delta_Lambda, x[1])]

    while ((max(abs(g(x))) > tol) and (m < M)):
        x[1] = x[1] -dgdx(x[0:1]).T.dot(Delta_Lambda)

        # m, Delta_Lambda = m+1, g(x[1:2])/sum(dgdx(x[1:2])*dgdx(x[0:1]), axis=1)
        m, Delta_Lambda = m+1, g(x[1:2])/((dgdx(x[1:2]).dot(dgdx(x[0:1]).T)).diagonal())
        # x[0] = x[1] # TODO: needs further justification
        if store: info.append((m, Delta_Lambda, x[1]))
        
    # print('%s' %m)
        
    assert m < M, "Nonlinear solver did not converge"
    
    
    if store:
        return x, info
    else:
        return x, m, g(x[1:2])

#%% Testing
from numpy import sin, cos, exp, linspace, pi, array
import matplotlib.pyplot as plt
    
def _g(x):
    return exp(-0.1*x**2)*sin(pi/2*x)

def _dg(x):
    return -2*0.1*x*exp(-0.1*x**2)*sin(pi/2*x) + \
           pi/2*exp(-0.1*x**2)*cos(pi/2*x)

def _test():
    import sys

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