from numpy import linalg as LA

def Newton(f, x, dfdx, epsilon=1.0E-7, N=100, store=False):
    f_value = f(x)
    n = 0
    if store: info = [(x, f_value, LA.norm(dfdx(x)))]
    while LA.norm(f_value) > epsilon and n <= N:
        dfdx_value = dfdx(x)
        if LA.norm(dfdx_value) < 1E-14:
            raise ValueError("Newton: f'(%g)=%g" % (x, LA.norm(dfdx_value)))

        try:
            x = x - LA.solve(dfdx_value, f_value)
        except LA.LinAlgError:
            x = x - f_value/dfdx_value
        except:
            raise LA.LinAlgError("Unable to solve the system")

        n += 1
        f_value = f(x)
        if store: info.append((x, f_value, dfdx_value))
    if store:
        return x, info
    else:
        return x, n, f_value

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