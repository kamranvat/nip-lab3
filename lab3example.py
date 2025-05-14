import numpy as np
import matplotlib.pyplot as plt


def euler_integrate(
    derivs,  # function
    x0,  # initial state
    t,  # time
):
    x = np.empty((len(t), len(x0)))

    x[0] = x0

    for k in range(len(t) - 1):
        dt = t[k + 1] - t[k]
        x[k + 1] = x[k] + dt * derivs(t[k], x[k])

    return x


def alpha_m(v):
    return 0.32 * (v + 54) / (1 - np.exp(-(v + 54) / 4))


def beta_m(v):
    return 0.28 * (v + 27) / (np.exp((v + 27) / 5) - 1)


def alpha_h(v):
    return 0.128 * np.exp(-(v + 50) / 18)


def beta_h(v):
    return 4 / (1 + np.exp(-(v + 27) / 5))


def alpha_n(v):
    return 0.032 * (v + 52) / (1 - np.exp(-(v + 52) / 5))


def beta_n(v):
    return 0.5 * np.exp(-(v + 57) / 40)


def calc_x_inf(alpha_x, beta_x):
    return alpha_x / (alpha_x + beta_x)


def calc_tau(alpha_x, beta_x):
    return 1 / (alpha_x + beta_x)


def calc_xdot(x, x_inf, tau):
    return -(1 / tau) * (x - x_inf)


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


def plot_trajectories(traj, t):
    # Extract variables
    v = traj[:, 0]
    m = traj[:, 1]
    n = traj[:, 2]
    h = traj[:, 3]

    # Plot results
    plt.figure(figsize=(10, 8))

    # Voltage trace
    plt.subplot(4, 1, 1)
    plt.plot(t, v, label="Voltage (mV)")
    plt.axhline(-55, color="r", linestyle="--", label="Firing Threshold")
    plt.title("Hodgkin-Huxley Neuron Simulation")
    plt.ylabel("Voltage (mV)")
    plt.legend()
    plt.grid()

    # Gating variables
    plt.subplot(4, 1, 2)
    plt.plot(t, m, label="m")
    plt.ylabel("m")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 3)
    plt.plot(t, n, label="n")
    plt.ylabel("n")
    plt.legend()
    plt.grid()

    plt.subplot(4, 1, 4)
    plt.plot(t, h, label="h")
    plt.xlabel("Time (ms)")
    plt.ylabel("h")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def HH_derivative(I):
    # I is i_c (the step current) over time
    C_m = 1.0
    g_na = 50.0
    g_k = 10.0
    g_l = 0.1
    E_na = 50.0
    E_k = -90.0
    E_l = -65.0

    def derivs(t, x):
        v, m, n, h = x

        # Steady values
        m_inf = calc_x_inf(alpha_m(v), beta_m(v))
        n_inf = calc_x_inf(alpha_n(v), beta_n(v))
        h_inf = calc_x_inf(alpha_h(v), beta_h(v))

        # Taus
        tau_m = calc_tau(alpha_m(v), beta_m(v))
        tau_n = calc_tau(alpha_n(v), beta_n(v))
        tau_h = calc_tau(alpha_h(v), beta_h(v))

        # Derivatives
        mdot = calc_xdot(m, m_inf, tau_m)
        ndot = calc_xdot(n, n_inf, tau_n)
        hdot = calc_xdot(h, h_inf, tau_h)

        # Currents
        i_na = g_na * m**3 * h * (v - E_na)
        i_k = g_k * n**4 * (v - E_k)
        i_l = g_l * (v - E_l)
        i_c = I(t)

        # Voltage change
        vdot = (i_c - i_na - i_k - i_l) / C_m

        return np.array([vdot, mdot, ndot, hdot])

    return derivs


def count_spikes(v_traj, thresh):
    """return nr of spikes in v_traj"""
    spikes = 0
    for i in range(len(v_traj) - 1):
        v = v_traj[i + 1]
        v_prev = v_traj[i]
        if v_prev < thresh and v > thresh:
            spikes += 1

    return spikes


def run_experiment(
    T=50, dt=0.025, amp=3.0, i_c_start=10.0, i_c_end=20.0, i_c_rest=0.0, v_0=-65.0
):
    t = np.arange(0.0, T + dt, dt)
    # step current:
    I = lambda t: amp if i_c_start <= t <= i_c_end else i_c_rest
    # initial state:
    x0 = np.array(
        [
            v_0,
            calc_x_inf(alpha_m(v_0), beta_m(v_0)),
            calc_x_inf(alpha_n(v_0), beta_n(v_0)),
            calc_x_inf(alpha_h(v_0), beta_h(v_0)),
        ]
    )

    # integrate the system:
    traj = euler_integrate(HH_derivative(I), x0, t)

    return traj, t


def plot_fi_curve(current, spike_count):
    plt.figure(figsize=(8, 6))
    plt.plot(current, spike_count, marker="o")
    plt.title("F-I Curve")
    plt.xlabel("Current (mA)")
    plt.ylabel("Spike Count")
    plt.grid()
    plt.show()


# Task 1, pt. 1 - run a single experiment, T = 50, amp = 3.0, i_c_start = 10.0, i_c_end = 20.0
print("Running single experiment...")
all_traces, t = run_experiment(
    T=50, dt=0.025, amp=3.0, i_c_start=10.0, i_c_end=20.0, i_c_rest=0.0, v_0=-65.0
)
plot_trajectories(all_traces, t)

# Task 1, pt. 2 - run 51 experiments:
exp_nr = 51
currents = np.linspace(0.0, 10.0, exp_nr)  # TODO check docs
onset = 50
cutoff = 200
v_traces = []
spike_counts = []
thresh = 0.0

print("Running multiple experiments for F-I-curve...")
for i in range(0, exp_nr):
    amp = currents[i]
    all_traces, t = run_experiment(T=250, amp=amp, i_c_start=onset, i_c_end=cutoff)
    if i % 10 == 0:
        print(i, "of", exp_nr, "...")
    v_trace = all_traces[:, 0]
    spike_counts.append(count_spikes(v_trace, thresh))

plot_fi_curve(currents, spike_counts)
