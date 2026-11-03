from math import fabs, sqrt
import numpy as np
import sys
import time

# 進捗表示：label を与えた solve1 / solve2 は，PROGRESS_INTERVAL 秒以上かかった場合に
# その間隔で「反復回数・ステップの最大相対変化（収束判定量）・残差」を表示する（速く終わる呼び出しは何も表示しない）
PROGRESS_INTERVAL = 10.0


class _Progress(object):
    def __init__(self, label, tol):
        self.label = label
        self.tol = tol
        self.t0 = time.time()
        self.last = self.t0
        self.printed = False
        self.it = 0

    def step(self, maxerr, h):
        self.it += 1
        if self.label is None:
            return
        now = time.time()
        if now - self.last >= PROGRESS_INTERVAL:
            self.last = now
            self.printed = True
            print(f"    [{self.label}] 反復 {self.it:,}，経過 {now - self.t0:,.0f} 秒，"
                  f"ステップの最大相対変化 {maxerr:.2e}（目標 {self.tol:.0e}），残差 0.5|f|^2 = {h:.3e}", flush=True)

    def done(self):
        if self.label is not None and self.printed:
            print(f"    [{self.label}] 収束：反復 {self.it:,}，{time.time() - self.t0:,.0f} 秒", flush=True)

class Optimize_result(object):
    def __init__(self, x, fun, jac_inv, hess_inv, hess):
        self.x = x
        self.fun = fun
        self.jac_inv = jac_inv
        self.hess_inv = hess_inv
        self.hess = hess

def hbasecalc():
    return sys.float_info.epsilon ** (1.0/3.0)

def line_search1(h0, hd0, h1, l1):
    b = hd0
    c = h0
    a = (h1 - b*l1 - c)/(l1*l1)
    if a != 0.0:
        l2 = -b/(2.0*a)
    else:
        l2 = 0.5*l1
    if l2 > 0.5*l1:
        l2 = 0.5*l1
    elif l2 < 0.1*l1:
        l2 = 0.1*l1
    return l2

def line_search2(h0, hd0, h1, h2, l1, l2):
    c = hd0
    d = h0
    det = l1*l1*l2*l2*(l1 - l2)
    a = ( l2*l2*(h1 - c*l1 - d) - l1*l1*(h2 - c*l2 - d) )/det
    b = ( -l2*l2*l2*(h1 - c*l1 - d) + l1*l1*l1*(h2 - c*l2 - d) )/det
    if a != 0.0 and b*b - 3.0*a*c >= 0.0:
        l3 = ( -b + sqrt(b*b - 3.0*a*c) )/(3.0*a)
    else:
        l3 = 0.5*l2
    if l3 > 0.5*l2:
        l3 = 0.5*l2
    elif l3 < 0.1*l2:
        l3 = 0.1*l2
    return l3

def numerical_jacobian(func, x, n, m, typx):
    hbase = hbasecalc()
    J = np.zeros((n,m))
    xs = x.copy()
    for j in range(m):
        h = hbase*max(fabs(x[j]), typx[j])
        xs[j] = x[j] + h
        fp = func(xs)
        xs[j] = x[j] - h
        fm = func(xs)
        for i in range(n):
            J[i,j] = (fp[i] - fm[i])/(2.0*h)
        xs[j] = x[j]
    return J

def solve1(func, x, jac = None, tol = 1.0e-13, typx = None, label = None):
    prog = _Progress(label, tol)
    n = x.size
    if typx is None:
        typx = np.zeros(n)
    xs = np.zeros(n)
    f = func(x)
    while True:
        if jac is None:
            J = numerical_jacobian(func, x, n, n, typx)
        else:
            J = jac(x)
        h0 = 0.5*np.dot(f, f)
        hd0 = -2.0*h0
        w = np.linalg.solve(J, f)
        maxerr = 0.0
        for i in range(n):
            err = fabs(w[i])/max(fabs(x[i]), typx[i])
            if err > maxerr:
                maxerr = err
        l1 = 1.0
        while True:
            xs[:] = x[:] - l1*w[:]
            try:
                f = func(xs)
                break
            except ValueError:
                l1 *= 0.5
        h1 = 0.5*np.dot(f, f)
        if h1 <= h0 or l1 < 1.0e-2/max(maxerr, 1.0):
            l3 = l1
        else:
            l2 = line_search1(h0, hd0, h1, l1)
            xs[:] = x[:] - l2*w[:]
            f = func(xs)
            h2 = 0.5*np.dot(f, f)
            if h2 <= h0 or l2 < 1.0e-2/max(maxerr, 1.0):
                l3 = l2
            else:
                while True:
                    l3 = line_search2(h0, hd0, h1, h2, l1, l2)
                    xs[:] = x[:] - l3*w[:]
                    f = func(xs)
                    h3 = 0.5*np.dot(f, f)
                    if h3 <= h0 or l3 < 1.0e-2/max(maxerr, 1.0):
                        break
                    else:
                        l1, l2 = l2, l3
                        h1, h2 = h2, h3
        x[:] = xs[:]
        w[:] *= -l3
        maxerr = 0.0
        for i in range(n):
            err = fabs(w[i])/max(fabs(x[i]), typx[i])
            if err > maxerr:
                maxerr = err
        prog.step(maxerr, 0.5*np.dot(f, f))
        if maxerr < tol:
            break
    prog.done()
    optimize_result = Optimize_result(x, f, np.linalg.inv(J), None, None)
    return optimize_result

def bsu(JI, w, y):
    JIy = np.dot(JI, y)
    denominator = np.dot(w, JIy)
    if denominator != 0.0:
        wTJI = np.dot(JI.T, w)
        JI[:,:] += np.outer(w - JIy, wTJI)/denominator

def solve2(func, x, JI = None, jac = None, tol = 1.0e-13, typx = None, label = None):
    prog = _Progress(label, tol)
    n = x.size
    if typx is None:
        typx = np.zeros(n)
    xs = np.zeros(n)
    failure_count = 0
    f = func(x)
    if JI is None:
        if jac is None:
            J = numerical_jacobian(func, x, n, n, typx)
        else:
            J = jac(x)
        JI = np.linalg.inv(J)
    h0 = 0.5*np.dot(f, f)
    min_h = h0
    hd0 = -2.0*h0
    while True:
        w = np.dot(JI, f)
        maxerr = 0.0
        for i in range(n):
            err = fabs(w[i])/max(fabs(x[i]), typx[i])
            if err > maxerr:
                maxerr = err
        l1 = 0.5/max(maxerr, 0.5)
        while True:
            xs[:] = x[:] - l1*w[:]
            try:
                fun2 = func(xs)
                break
            except ValueError:
                l1 *= 0.5
        h1 = 0.5*np.dot(fun2, fun2)
        if h1 <= h0 or l1 < 1.0e-2/max(maxerr, 1.0):
            l3 = l1
            h3 = h1
        else:
            l2 = line_search1(h0, hd0, h1, l1)
            xs[:] = x[:] - l2*w[:]
            fun2 = func(xs)
            h2 = 0.5*np.dot(fun2, fun2)
            if h2 <= h0 or l2 < 1.0e-2/max(maxerr, 1.0):
                l3 = l2
                h3 = h2
            else:
                while True:
                    l3 = line_search2(h0, hd0, h1, h2, l1, l2)
                    xs[:] = x[:] - l3*w[:]
                    fun2 = func(xs)
                    h3 = 0.5*np.dot(fun2, fun2)
                    if h3 <= h0 or l3 < 1.0e-2/max(maxerr, 1.0):
                        break
                    else:
                        l1, l2 = l2, l3
                        h1, h2 = h2, h3
        if h3 <= min_h:
            failure_count = 0
            min_h = h3
        else:
            failure_count += 1
        h0 = h3
        hd0 = -2.0*h0
        x[:] = xs[:]
        w[:] *= -l3
        maxerr = 0.0
        for i in range(n):
            err = fabs(w[i])/max(fabs(x[i]), typx[i])
            if err > maxerr:
                maxerr = err
        prog.step(maxerr, h3)
        if maxerr < tol:
            break
        if failure_count < 100:
            bsu(JI, w, fun2 - f)
        else:
            if jac is None:
                J = numerical_jacobian(func, x, n, n, typx)
            else:
                J = jac(x)
            JI = np.linalg.inv(J)
            failure_count = 0
        f[:] = fun2[:]
    prog.done()
    optimize_result = Optimize_result(x, f, JI, None, None)
    return optimize_result

def numerical_gradient(func, x, typx):
    n = x.size
    hbase = hbasecalc()
    g = np.zeros(n)
    xs = x.copy()
    for i in range(n):
        h = hbase*max(fabs(x[i]), typx[i])
        xs[i] = x[i] + h
        fp = func(xs)
        xs[i] = x[i] - h
        fm = func(xs)
        g[i] = (fp - fm)/(2.0*h)
        xs[i] = x[i]
    return g

def bfgs(HI, w, z):
    wTz = np.dot(w, z)
    HIz = np.dot(HI, z)
    zTHIz = np.dot(z, HIz)
    if wTz > 0.0 and zTHIz > 0.0:
        u = w/wTz - HIz/zTHIz
        HI[:,:] += np.outer(w, w)/wTz - np.outer(HIz, HIz)/zTHIz + zTHIz*np.outer(u, u)

def optimize(func, x, HI = None, grad = None, tol1 = 1.0e-7, tol2 = 1.0e-2, typx = None):
    n = x.size
    if typx is None:
        typx = np.zeros(n)
    xs = np.zeros(n)
    failure_count = 0
    failure_count2 = 0
    h0 = func(x)
    min_fun = h0
    min_x = x.copy()
    if grad is None:
        g = numerical_gradient(func, x, typx)
    else:
        g = grad(x)
    if HI is None:
        HI = np.identity(n)
    while True:
        w = np.dot(HI, g)
        hd0 = -np.dot(g, w)
        maxerr = 0.0
        for i in range(n):
            err = fabs(w[i])/max(fabs(x[i]), typx[i])
            if err > maxerr:
                maxerr = err
        l1 = 0.5/max(maxerr, 0.5)
        while True:
            xs[:] = x[:] - l1*w[:]
            try:
                h1 = func(xs)
                break
            except ValueError:
                l1 *= 0.5
        if h1 <= h0 or l1 < 1.0e-2/max(maxerr, 1.0):
            l3 = l1
            h3 = h1
        else:
            l2 = line_search1(h0, hd0, h1, l1)
            while True:
                xs[:] = x[:] - l2*w[:]
                try:
                    h2 = func(xs)
                    break
                except ValueError:
                    l2 *= 0.5
            if h2 <= h0 or l2 < 1.0e-2/max(maxerr, 1.0):
                l3 = l2
                h3 = h2
            else:
                while True:
                    l3 = line_search2(h0, hd0, h1, h2, l1, l2)
                    while True:
                        xs[:] = x[:] - l3*w[:]
                        try:
                            h3 = func(xs)
                            break
                        except ValueError:
                            l3 *= 0.5
                    if h3 <= h0 or l3 < 1.0e-2/max(maxerr, 1.0):
                        break
                    else:
                        l1, l2 = l2, l3
                        h1, h2 = h2, h3
        if h3 > h0:
            failure_count += 1
        h0 = h3
        x[:] = xs[:]
        w[:] *= -l3
        if h0 < min_fun:
            min_fun = h0
            min_x[:] = x[:]
        maxerr = 0.0
        for i in range(n):
            err = fabs(w[i])/max(fabs(x[i]), typx[i])
            if err > maxerr:
                maxerr = err
        maxerr2 = 0.0
        for i in range(n):
            err = fabs(g[i])*max(fabs(x[i]), typx[i])/max(fabs(h0), 1.0)
            if err > maxerr2:
                maxerr2 = err
        if maxerr < tol1 and maxerr2 < tol2:
            break
        if grad is None:
            g2 = numerical_gradient(func, x, typx)
        else:
            g2 = grad(x)
        if failure_count < 33:
            bfgs(HI, w, g2 - g)
        else:
            if failure_count2 < 3:
                HI[:,:] = np.identity(n)
                failure_count = 0
                failure_count2 += 1
            else:
                break
        g[:] = g2[:]
    optimize_result = Optimize_result(min_x, h0, None, HI, None)
    return optimize_result

def RMSprop(func, x, grad = None, learning_rate = None, epsilon = 1.0e-7, improve_limit = 0.0):
    xs = np.zeros_like(x)
    v = np.zeros_like(x)
    w = np.zeros_like(x)
    if learning_rate is None:
        learning_rate = 1.0e-1*np.ones_like(x)
    fun = func(x)
    min_fun = fun
    min_x = x.copy()
    failure_count = 0
    while True:
        if grad is None:
            g = numerical_gradient(func, x, learning_rate)
        else:
            g = grad(x)
        v[:] = 0.1*np.power(g[:], 2) + 0.9*v[:]
        w[:] = learning_rate[:]/np.sqrt(v[:] + epsilon)*g[:]
        l = 1.0
        while True:
            xs[:] = x[:] - l*w[:]
            try:
                fun = func(xs)
                break
            except ValueError:
                l *= 0.5
        x[:] = xs[:]
        if fun < min_fun - improve_limit:
            failure_count = 0
        else:
            failure_count += 1
        if fun < min_fun:
            min_fun = fun
            min_x[:] = x[:]
        if failure_count == 2:
            break
    return min_x

def modified_bfgs(B, s, y):
    Bs = np.dot(B, s)
    sTBs = np.dot(s, Bs)
    sTy = np.dot(s, y)
    if sTy < 0.5*sTBs:
        beta = (1.0 - 0.5)*sTBs/(sTBs - sTy)
        y = beta*y + (1.0 - beta)*Bs
        sTy = np.dot(s, y)
    B[:,:] += - np.outer(Bs, Bs)/sTBs + np.outer(y, y)/sTy

def optimize_constrained(func, cons, conse, x, B = None, grad = None, jac = None, jace = None, tol1 = 1.0e-11, tol2 = 1.0e-2, typx = None):
    mu = 0.1
    nu = 1.0
    theta = 0.1
    n = x.size
    if typx is None:
        typx = np.zeros(n)
    failure_count = 0
    func(x)
    if cons is None:
        m = 0
    else:
        m = cons(x).size
    if conse is None:
        me = 0
    else:
        me = conse(x).size
    xs = np.zeros(n)
    dx = np.zeros(n)
    y = np.zeros(n)
    ss = np.zeros(m)
    lams = np.zeros(m)
    ds = np.zeros(m)
    dlam = np.zeros(m)
    dlame = np.zeros(me)
    dL = np.zeros(n+2*m+me)
    HL = np.zeros((n+2*m+me,n+2*m+me))
    s = mu*np.ones(m)
    lam = np.ones(m)
    lame = np.zeros(me)
    f = func(x)
    g = np.zeros(m)
    if m > 0:
        g = cons(x)
    ge = np.zeros(me)
    if me > 0:
        ge = conse(x)
    if grad is None:
        df = numerical_gradient(func, x, typx)
    else:
        df = grad(x)
    dg = np.zeros((m,n))
    if m > 0:
        if jac is None:
            dg = numerical_jacobian(cons, x, m, n)
        else:
            dg = jac(x)
    dge = np.zeros((me,n))
    if me > 0:
        if jace is None:
            dge = numerical_jacobian(conse, x, me, n)
        else:
            dge = jace(x)
    for i in range(n):
        dL[i] = df[i] - np.dot(lam[0:m], dg[0:m,i]) - np.dot(lame[0:me], dge[0:me,i])
    dL[n:n+m] = lam[0:m]*s[0:m] - mu
    dL[n+m:n+2*m] = -g[0:m] + s[0:m]
    dL[n+2*m:n+2*m+me] = -ge[0:me]
    if B is None:
        B = np.identity(n)
    while True:
        HL[0:n,0:n] = B[0:n,0:n]
        HL[0:n,n+m:n+2*m] = -dg[0:m,0:n].T
        HL[0:n,n+2*m:n+2*m+me] = -dge[0:me,0:n].T
        HL[n:n+m,n:n+m] = np.diag(lam[0:m])
        HL[n:n+m,n+m:n+2*m] = np.diag(s[0:m])
        HL[n+m:n+2*m,0:n] = -dg[0:m,0:n]
        HL[n+m:n+2*m,n:n+m] = np.identity(m)
        HL[n+2*m:n+2*m+me,0:n] = -dge[0:me,0:n]
        dxslam = np.linalg.solve(HL, -dL)
        dx[0:n] = dxslam[0:n]
        ds[0:m] = dxslam[n:n+m]
        dlam[0:m] = dxslam[n+m:n+2*m]
        dlame[0:me] = dxslam[n+2*m:n+2*m+me]
        df_mu_logs_d = np.dot(df, dx) - np.sum(mu/s*ds)
        sum_squared = np.dot(g - s, g - s) + np.dot(ge, ge)
        dg_d = np.dot(dg, dx)
        dge_d = np.dot(dge, dx)
        dsum_squared_d = np.dot(g - s, dg_d - ds) + np.dot(ge, dge_d)
        if sum_squared > 0.0:
            dsum_squared_d /= sqrt(sum_squared)
        if df_mu_logs_d > 0.0 and dsum_squared_d < 0.0:
            nut = - df_mu_logs_d/dsum_squared_d/0.7
            if nut > nu:
                nu = nut
        sumlogs = np.sum(np.log(s))
        h0 = f - mu*sumlogs + nu*sqrt(sum_squared)
        hd0 = df_mu_logs_d + nu*dsum_squared_d
        maxerr = 0.0
        for i in range(n):
            err = fabs(dx[i])/max(fabs(x[i]), typx[i])
            if err > maxerr:
                maxerr = err
        l1 = 1.0
        if m > 0:
            while True:
                ss[:] = s[:] + l1*ds[:]
                if np.all(ss > 0.0):
                    break
                else:
                    l1 *= 0.5
            while True:
                lams[:] = lam[:] + l1*dlam[:]
                if np.all(lams > 0.0):
                    break
                else:
                    l1 *= 0.5
        while True:
            xs[:] = x[:] + l1*dx[:]
            try:
                h1 = func(xs)
                break
            except ValueError:
                l1 *= 0.5
        if m > 0:
            g = cons(xs)
            ss[:] = s[:] + l1*ds[:]
        if me > 0:
            ge = conse(xs)
        sum_squared = np.dot(g - ss, g - ss) + np.dot(ge, ge)
        sumlogs = np.sum(np.log(ss))
        h1 = f - mu*sumlogs + nu*sqrt(sum_squared)
        if h1 <= h0 or l1 < 1.0e-2/max(maxerr, 1.0):
            l3 = l1
            h3 = h1
        else:
            l2 = line_search1(h0, hd0, h1, l1)
            xs[:] = x[:] + l2*dx[:]
            f = func(xs)
            if m > 0:
                g = cons(xs)
                ss[:] = s[:] + l2*ds[:]
            if me > 0:
                ge = conse(xs)
            sum_squared = np.dot(g - ss, g - ss) + np.dot(ge, ge)
            sumlogs = np.sum(np.log(ss))
            h2 = f - mu*sumlogs + nu*sqrt(sum_squared)
            if h2 <= h0 or l2 < 1.0e-2/max(maxerr, 1.0):
                l3 = l2
                h3 = h2
            else:
                while True:
                    l3 = line_search2(h0, hd0, h1, h2, l1, l2)
                    xs[:] = x[:] + l3*dx[:]
                    f = func(xs)
                    if m > 0:
                        g = cons(xs)
                        ss[:] = s[:] + l3*ds[:]
                    if me > 0:
                        ge = conse(xs)
                    sum_squared = np.dot(g - ss, g - ss) + np.dot(ge, ge)
                    sumlogs = np.sum(np.log(ss))
                    h3 = f - mu*sumlogs + nu*sqrt(sum_squared)
                    if h3 <= h0 or l3 < 1.0e-2/max(maxerr, 1.0):
                        break
                    else:
                        l1, l2 = l2, l3
                        h1, h2 = h2, h3
        if h3 <= h0:
            failure_count = 0
        else:
            failure_count += 1
        x[:] = xs[:]
        s[:] = ss[:]
        lam[:] += l3*dlam[:]
        lame[:] += l3*dlame[:]
        dx[:] *= l3
        maxerr = 0.0
        for i in range(n):
            err = fabs(dx[i])/max(fabs(x[i]), typx[i])
            if err > maxerr:
                maxerr = err
        for i in range(n):
            y[i] = df[i] - np.dot(lam[0:m], dg[0:m,i]) - np.dot(lame[0:me], dge[0:me,i])
        if grad is None:
            df = numerical_gradient(func, x, typx)
        else:
            df = grad(x)
        if m > 0:
            if jac is None:
                dg = numerical_jacobian(cons, x, m, n)
            else:
                dg = jac(x)
        if me > 0:
            if jace is None:
                dge = numerical_jacobian(conse, x, me, n)
            else:
                dge = jace(x)
        for i in range(n):
            dL[i] = df[i] - np.dot(lam[0:m], dg[0:m,i]) - np.dot(lame[0:me], dge[0:me,i])
        dL[n:n+m] = lam[0:m]*s[0:m] - mu
        dL[n+m:n+2*m] = -g[0:m] + s[0:m]
        dL[n+2*m:n+2*m+me] = -ge[0:me]
        maxerr2 = 0.0
        for i in range(n+2*m+me):
            err = fabs(dL[i])/max(fabs(f), 1.0)
            if err > maxerr2:
                maxerr2 = err
        if maxerr < tol1 and maxerr2 < tol2:
            break
        if m > 0 and maxerr2 < theta:
            mu *= 0.2
            theta *= 0.2
            nu = 1.0
        if failure_count < 10:
            y[0:n] = dL[0:n] - y[0:n]
            modified_bfgs(B, dx, y)
        else:
            B[:,:] = np.identity(n)
            failure_count = 0
    if m > 0 and me > 0:
        optimize_result = Optimize_result(x, (f, g, ge), None, None, B)
    elif m > 0:
        optimize_result = Optimize_result(x, (f, g), None, None, B)
    elif me > 0:
        optimize_result = Optimize_result(x, (f, ge), None, None, B)
    else:
        optimize_result = Optimize_result(x, f, None, None, B)
    return optimize_result
