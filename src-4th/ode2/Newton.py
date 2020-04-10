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


def _g(x):
    from scitools.std import exp, sin, pi
    return exp(-0.1*x**2)*sin(pi/2*x)

def _dg(x):
    from scitools.std import exp, sin, cos, pi
    return -2*0.1*x*exp(-0.1*x**2)*sin(pi/2*x) + \
           pi/2*exp(-0.1*x**2)*cos(pi/2*x)

def _test():
    from scitools.std import sin, cos, exp, linspace, plot, pi
    import sys

    #x0 = float(sys.argv[1])
    x0 = 0.1
    x, info = Newton(_g, x0, _dg, store=True)
    print 'root: %.16g' % x
    for i in range(len(info)):
        print 'Iteration %2d: f(%g)=%g, dF(%g) =%g' % \
              (i, info[i][0], info[i][1], info[i][0], info[i][2])

    x = linspace(-7, 7, 401)
    y = _g(x)
    plot(x, y, 'b.', xlabel='x', ylabel='y',
     title="Newton Iterates", savefig='tmp.pdf')

if __name__ == '__main__':
    _test()