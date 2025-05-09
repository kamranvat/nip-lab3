import numpy as np
import matplotlib.pyplot as plt


def euler_integrate(
    derivs, # function
    x0, # initial state
    t, # time
):
    x = np.empty((len(t), len(x0)))

    x[0] = x0

    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        x[k + 1] = x[k] + dt * derivs(t[k], x[k])

    return x


def RC_derivative(tau, I):
    """f(x,t)"""

    def deriv(t, x):
        dx = -1 / tau * x + I(t)  # simplest possible differential equation
        return np.array([dx])

    return deriv


def calc_x_inf(alpha_x, beta_x):
    # steady state value of x:
    x = alpha_x / (alpha_x + beta_x)
    return x


def calc_tau(alpha_x:float, beta_x:float):
    return 1 / (alpha_x + beta_x)


def calc_xdot(x, x_inf, tau):
    # calculate dx/dx x:
    return -(1 / tau) * (x - x_inf)


def alpha_n(v:float):
    return 0.032 * ((v + 52) / 1 - np.exp(-(v + 52) / 50))


def beta_n(v:float):
    return 0.5 * np.exp(-(v + 57) / 40)


def alpha_m(v:float):
    alpha_m = 0.32 * ((v + 54) / 1 - np.exp(-(v + 54) / 4))
    if type(v) == float:
        print(v)

    if type(alpha_m) == float:
        print(alpha_m)

    return alpha_m


def beta_m(v:float):
    return 0.28 * ((v + 27) / np.exp((v + 27) / 5) - 1)


def alpha_h(v:float):
    return 0.128 * np.exp(-(v + 50) / 18)


def beta_h(v:float):
    return 4 / (1 + np.exp(-(v + 27) / 5))


def gating_fct_na(g_na, m, h, v, E_na):
    # sodium current:
    i_na = g_na * m**3 * h * (v - E_na)
    return i_na


def gating_fct_k(g_k, n, v, E_k):
    # potassium current:
    i_k = g_k * n**4 * (v - E_k)
    return i_k


def gating_fct_l(g_l, v, E_l):
    # leak current:
    i_l = g_l * (v - E_l)
    return i_l


def HH_derivative(tau, I):
    # I is I_c (the step current)
    # c_m dv/dt = i_c - i_na - i_k - i_l
    # all the parameters:
    C_m = 1.0
    g_na = 50.0
    g_k = 10.0
    g_l = 0.1
    E_na = 50.0
    E_k = -90.0
    E_l = -65.0



    # get infs from current voltage
    m_inf = lambda v: calc_x_inf(alpha_m(v), beta_m(v))
    n_inf = lambda v: calc_x_inf(alpha_n(v), beta_n(v))
    h_inf = lambda v: calc_x_inf(alpha_h(v), beta_h(v))

    # tau with current voltage
    tau_m = lambda v: calc_tau(alpha_m(v), beta_m(v))
    tau_n = lambda v: calc_tau(alpha_n(v), beta_n(v))
    tau_h = lambda v: calc_tau(alpha_h(v), beta_h(v))

    def derivs(t, x):
        # unpack x into V, m, n, h:
        v, m, n, h = x

        # do all gate updates (the calc_xdot function)
        mdot = calc_xdot(m, m_inf(v), tau_m(v))
        ndot = calc_xdot(n, n_inf(v), tau_n(v))
        hdot = calc_xdot(h, h_inf(v), tau_h(v))

        # calculate all currents (gating_fct_na, gating_fct_k, gating_fct_l)
        i_na = gating_fct_na(g_na, m, h, v, E_na)
        i_k = gating_fct_k(g_k, n, v, E_k)
        i_l = gating_fct_l(g_l, v, E_l)
        i_c = I(t)

        # then calculate dv_dt: I(t) - i_na - i_k - i_l / C_m
        vdot = (i_c - i_na - i_k - i_l) / C_m

        return np.array([vdot, mdot, ndot, hdot])

    return derivs


def plot_trajectory(t, x, title, ylab=""):
    plt.figure()
    plt.plot(t, x)
    plt.title(title)
    plt.xlabel("time (ms)")
    plt.ylabel(ylab)
    plt.grid()
    plt.show()


dt = 0.1
T = 50
t = np.arange(0.0, T + dt, dt)
tau = 20
amp = 3.0
# step current:
I = lambda t: amp if 10.0 <= t <= 20.0 else 0.0
# initial gating at steady state for v0:
x0 = np.array(
    [
        -65.0,
        calc_x_inf(alpha_m(-65.0), beta_m(-65.0)),
        calc_x_inf(alpha_n(-65.0), beta_n(-65.0)),
        calc_x_inf(alpha_h(-65.0), beta_h(-65.0)),
    ]
)

# integrate the system:
# RC = RC_derivative(tau, I)
# HH = HH_derivative(tau, x0, I)
traj = euler_integrate(HH_derivative(tau, I), x0, t)

# plot the results:
plot_trajectory(t, traj, "RC circuit response to step current", "Voltage (V)")
