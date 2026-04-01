import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg



def simulate_mm1(lambda_rate, mu_rate, N, seed):
 
    if lambda_rate <= 0 or mu_rate <= 0:
        raise ValueError("lambda_rate and mu_rate must be positive.")
    if lambda_rate >= mu_rate:
        raise ValueError("Need lambda_rate < mu_rate for a stable M/M/1 queue.")

    rng = np.random.default_rng(seed)

    # Generate random variables
    interarrivals = rng.exponential(scale=1 / lambda_rate, size=N)
    services = rng.exponential(scale=1 / mu_rate, size=N)

    # Arrival times
    arrivals = np.cumsum(interarrivals)

    # State arrays
    wait_q = np.zeros(N)
    start_service = np.zeros(N)
    departures = np.zeros(N)
    system_times = np.zeros(N)

    # First customer
    start_service[0] = arrivals[0]
    departures[0] = start_service[0] + services[0]
    system_times[0] = services[0]

    # Remaining customers
    for i in range(1, N):
        wait_q[i] = max(0.0, departures[i - 1] - arrivals[i])  ## Lindley's
        start_service[i] = arrivals[i] + wait_q[i]
        departures[i] = start_service[i] + services[i]
        system_times[i] = wait_q[i] + services[i]

    return {
        "interarrivals": interarrivals,
        "services": services,
        "arrivals": arrivals,
        "wait_q": wait_q,
        "start_service": start_service,
        "departures": departures,
        "system_times": system_times
    }


"""## IPA estimator with respect to the arrival rate $\mu$

$$W_i = \max\{0,\;W_{i-1}+S_{i-1}-A_i\}$$

So on sample paths where the inside of the max is positive,

$$D_i^{(\mu)}=D_{i-1}^{(μ)}+\frac{dS_{i-1}}{dμ}$$

If system becomes empty before customer $i$, the pertubation resets to zero:
$$D_i^{(\mu)}=\begin{cases}D_{i-1}^{(\mu)} - \dfrac{S_{i-1}}{\mu}, & \text{if } W_{i-1}+S_{i-1}-A_i > 0,\\[1em]
0, & \text{otherwise.}\end{cases}$$

Since
$$T_i = W_i+S_i,$$

we get

$$\frac{dT_i}{d\mu}=D_i^{(\mu)}-\frac{S_i}{\mu}$$

Therefore IPA estimator is
$$\boxed{
\left[\frac{dJ}{d\mu}\right]_{\text{IPA}}
=
\frac{1}{N}\sum_{i=1}^N \left(D_i^{(\mu)} - \frac{S_i}{\mu}\right)
}$$

## IPA estimator with respect to the arrival rate $\lambda$

Write interarrival times as
$$
A_i = \frac{-\ln V_i}{\lambda}.
$$
Then
$$
\frac{dA_i}{d\lambda} = -\frac{A_i}{\lambda}.
$$

Let
$$
D_i^{(\lambda)} = \frac{dW_i}{d\lambda}.
$$

Again from Lindley's recursion, when the queue remains busy,
$$
D_i^{(\lambda)}
= D_{i-1}^{(\lambda)} - \frac{dA_i}{d\lambda}
= D_{i-1}^{(\lambda)} + \frac{A_i}{\lambda}.
$$
If the system empties before customer $i$, the perturbation is reset:
$$
D_i^{(\lambda)} =
\begin{cases}
D_{i-1}^{(\lambda)} + \dfrac{A_i}{\lambda}, & \text{if } W_{i-1}+S_{i-1}-A_i > 0,\\[1em]
0, & \text{otherwise.}
\end{cases}
$$

Since service times do not depend on $\lambda$,

$$\frac{dT_i}{d\lambda} = D_i^{(\lambda)}.$$


Thus
$$
\boxed{
\left[\frac{dJ}{d\lambda}\right]_{\text{IPA}}
=
\frac{1}{N}\sum_{i=1}^N D_i^{(\lambda)}
}
$$

### Interpretation

A perturbation in an interarrival time affects future waiting times only if it occurs during a busy period. Once the system empties, that perturbation no longer propagates.
"""


def ipa_estimators_mm1(lambda_rate, mu_rate, N, seed):
    """
    Simulate M/M/1 and compute IPA derivative estimators for:
      - dJ/dmu
      - dJ/dlambda
    """

    rng = np.random.default_rng(seed)

    # Use inverse transform so dependence on lambda and mu is explicit
    U_arr = rng.uniform(size=N)
    U_srv = rng.uniform(size=N)

    interarrivals = -np.log(U_arr) / lambda_rate
    services = -np.log(U_srv) / mu_rate

    arrivals = np.cumsum(interarrivals)

    wait_q = np.zeros(N)
    departures = np.zeros(N)
    system_times = np.zeros(N)

    # IPA derivative states
    dW_dmu = np.zeros(N)
    dW_dlambda = np.zeros(N)
    dT_dmu = np.zeros(N)
    dT_dlambda = np.zeros(N)

    # first customer
    departures[0] = arrivals[0] + services[0]
    system_times[0] = services[0]

    # derivatives for first customer
    dT_dmu[0] = -services[0] / mu_rate
    dT_dlambda[0] = 0.0

    for i in range(1, N):
        x = wait_q[i - 1] + services[i - 1] - interarrivals[i]

        if x > 0:
            wait_q[i] = x
            dW_dmu[i] = dW_dmu[i - 1] - services[i - 1] / mu_rate
            dW_dlambda[i] = dW_dlambda[i - 1] + interarrivals[i] / lambda_rate
        else:
            wait_q[i] = 0.0
            dW_dmu[i] = 0.0
            dW_dlambda[i] = 0.0

        departures[i] = arrivals[i] + wait_q[i] + services[i]
        system_times[i] = wait_q[i] + services[i]

        dT_dmu[i] = dW_dmu[i] - services[i] / mu_rate
        dT_dlambda[i] = dW_dlambda[i]

    # running estimators
    running_J = np.cumsum(system_times) / np.arange(1, N + 1)
    running_dJ_dmu = np.cumsum(dT_dmu) / np.arange(1, N + 1)
    running_dJ_dlambda = np.cumsum(dT_dlambda) / np.arange(1, N + 1)

    return {
        "interarrivals": interarrivals,
        "services": services,
        "arrivals": arrivals,
        "wait_q": wait_q,
        "departures": departures,
        "system_times": system_times,
        "dT_dmu": dT_dmu,
        "dT_dlambda": dT_dlambda,
        "running_J": running_J,
        "running_dJ_dmu": running_dJ_dmu,
        "running_dJ_dlambda": running_dJ_dlambda
    }


def clear_plot_frame():
    for widget in plot_frame.winfo_children():
        widget.destroy()


def add_plot(parent, row, col, xdata, ydata, theory_value, xlabel, ylabel, title, line_label, theory_label):
    fig = Figure(figsize=(3.6, 2.6), dpi=100)
    ax = fig.add_subplot(111)

    ax.plot(xdata, ydata, linewidth=1.2, label=line_label)
    ax.axhline(theory_value, linestyle="--", linewidth=1.2, label=theory_label)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(True)
    ax.legend(fontsize=8)
    fig.tight_layout()

    canvas_plot = FigureCanvasTkAgg(fig, master=parent)
    canvas_plot.draw()
    canvas_plot.get_tk_widget().grid(row=row, column=col, padx=8, pady=8, sticky="nsew")


def update_scrollregion(event=None):
    canvas.configure(scrollregion=canvas.bbox("all"))


def _on_mousewheel(event):
    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


def run_gui_simulation():
    try:
        # User inputs
        lambda_rate = float(entry_lambda.get())
        mu_rate = float(entry_mu.get())
        N = int(entry_N.get())
        seed = int(entry_seed.get())

        if lambda_rate >= mu_rate:
            raise ValueError("System unstable: need lambda < mu.")

        """## Simulation and basic plots"""

        sim = simulate_mm1(lambda_rate, mu_rate, N, seed)
        arrivals = sim["arrivals"]
        departures = sim["departures"]
        wait_q = sim["wait_q"]
        system_times = sim["system_times"]
        services = sim["services"]

        running_mean_T = np.cumsum(system_times) / np.arange(1, N + 1)

        out = ipa_estimators_mm1(lambda_rate=lambda_rate, mu_rate=mu_rate, N=N, seed=seed)

        theory_J = 1 / (mu_rate - lambda_rate)
        theory_dJ_dmu = -1 / (mu_rate - lambda_rate) ** 2
        theory_dJ_dlambda = 1 / (mu_rate - lambda_rate) ** 2

        mean_est_var.set(f"{out['running_J'][-1]:.6f}")
        mean_theory_var.set(f"{theory_J:.6f}")
        mean_error_var.set(f"{abs(out['running_J'][-1] - theory_J):.6f}")

        dmu_est_var.set(f"{out['running_dJ_dmu'][-1]:.6f}")
        dmu_theory_var.set(f"{theory_dJ_dmu:.6f}")
        dmu_error_var.set(f"{abs(out['running_dJ_dmu'][-1] - theory_dJ_dmu):.6f}")

        dlambda_est_var.set(f"{out['running_dJ_dlambda'][-1]:.6f}")
        dlambda_theory_var.set(f"{theory_dJ_dlambda:.6f}")
        dlambda_error_var.set(f"{abs(out['running_dJ_dlambda'][-1] - theory_dJ_dlambda):.6f}")

        sample_avg_var.set(f"{np.mean(system_times):.6f}")
        sample_theory_var.set(f"{1 / (mu_rate - lambda_rate):.6f}")

        clear_plot_frame()

        for i in range(3):
            plot_frame.grid_columnconfigure(i, weight=1)

        add_plot(
            plot_frame, 0, 0,
            np.arange(1, N + 1),
            running_mean_T,
            theory_J,
            "Number of customers",
            "Mean system time",
            "Convergence of Mean System Time",
            "Running estimate of mean system time",
            "Theoretical mean"
        )

        add_plot(
            plot_frame, 0, 1,
            np.arange(1, N + 1),
            out["running_dJ_dmu"],
            theory_dJ_dmu,
            "Number of customers",
            "dJ/dmu",
            "Convergence of IPA estimate for dJ/dmu",
            "IPA estimate",
            "Theoretical estimate"
        )

        add_plot(
            plot_frame, 0, 2,
            np.arange(1, N + 1),
            out["running_dJ_dlambda"],
            theory_dJ_dlambda,
            "Number of customers",
            "dJ/dlambda",
            "Convergence of IPA estimate for dJ/dlambda",
            "IPA estimate",
            "Theoretical estimate"
        )

        root.after(100, update_scrollregion)

    except ValueError as e:
        messagebox.showerror("Input Error", str(e))


# --------------------------------------------------
# Main window
# --------------------------------------------------
root = tk.Tk()
root.title("IPA M/M/1 Simulator")
root.geometry("1280x760")
root.configure(bg="#f2f4f7")

style = ttk.Style()
style.theme_use("clam")

style.configure("Card.TFrame", background="white")
style.configure("CardTitle.TLabel", background="white", font=("Segoe UI", 12, "bold"))
style.configure("Body.TLabel", background="white", font=("Segoe UI", 10))
style.configure("Value.TLabel", background="white", font=("Segoe UI", 10, "bold"))
style.configure("Run.TButton", font=("Segoe UI", 10, "bold"), padding=6)

# --------------------------------------------------
# Scrollable container
# --------------------------------------------------
outer_frame = tk.Frame(root, bg="#f2f4f7")
outer_frame.pack(fill="both", expand=True)

canvas = tk.Canvas(outer_frame, bg="#f2f4f7", highlightthickness=0)
v_scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
h_scrollbar = ttk.Scrollbar(root, orient="horizontal", command=canvas.xview)

canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

v_scrollbar.pack(side="right", fill="y")
h_scrollbar.pack(side="bottom", fill="x")
canvas.pack(side="left", fill="both", expand=True)

main_frame = ttk.Frame(canvas)
canvas_window = canvas.create_window((0, 0), window=main_frame, anchor="nw")

def resize_canvas_window(event):
    canvas.itemconfig(canvas_window, width=event.width)

canvas.bind("<Configure>", resize_canvas_window)
main_frame.bind("<Configure>", update_scrollregion)
canvas.bind_all("<MouseWheel>", _on_mousewheel)

# --------------------------------------------------
# Input card
# --------------------------------------------------
input_card = ttk.Frame(main_frame, style="Card.TFrame", padding=16)
input_card.pack(fill="x", padx=16, pady=(16, 12))

ttk.Label(input_card, text="User inputs", style="CardTitle.TLabel").grid(
    row=0, column=0, columnspan=4, sticky="w", pady=(0, 12)
)

ttk.Label(input_card, text="Arrival rate (lambda)", style="Body.TLabel").grid(row=1, column=0, padx=8, pady=6, sticky="w")
entry_lambda = ttk.Entry(input_card, width=14)
entry_lambda.grid(row=1, column=1, padx=8, pady=6, sticky="w")
entry_lambda.insert(0, "0.8")

ttk.Label(input_card, text="Service rate (mu)", style="Body.TLabel").grid(row=1, column=2, padx=8, pady=6, sticky="w")
entry_mu = ttk.Entry(input_card, width=14)
entry_mu.grid(row=1, column=3, padx=8, pady=6, sticky="w")
entry_mu.insert(0, "1.2")

ttk.Label(input_card, text="Number of customers (simulation length)", style="Body.TLabel").grid(row=2, column=0, padx=8, pady=6, sticky="w")
entry_N = ttk.Entry(input_card, width=14)
entry_N.grid(row=2, column=1, padx=8, pady=6, sticky="w")
entry_N.insert(0, "10000")

ttk.Label(input_card, text="Random seed", style="Body.TLabel").grid(row=2, column=2, padx=8, pady=6, sticky="w")
entry_seed = ttk.Entry(input_card, width=14)
entry_seed.grid(row=2, column=3, padx=8, pady=6, sticky="w")
entry_seed.insert(0, "40")

ttk.Button(input_card, text="Run Simulation", style="Run.TButton", command=run_gui_simulation).grid(
    row=3, column=0, columnspan=4, pady=(12, 0)
)

# --------------------------------------------------
# Results card
# --------------------------------------------------
results_card = ttk.Frame(main_frame, style="Card.TFrame", padding=16)
results_card.pack(fill="x", padx=16, pady=(0, 12))

ttk.Label(results_card, text="Final simulation estimates", style="CardTitle.TLabel").grid(
    row=0, column=0, columnspan=4, sticky="w", pady=(0, 12)
)

headers = ["Metric", "Estimate", "Theory", "Absolute Error"]
for j, h in enumerate(headers):
    ttk.Label(results_card, text=h, style="Value.TLabel").grid(row=1, column=j, padx=20, pady=4, sticky="w")

sample_avg_var = tk.StringVar(value="—")
sample_theory_var = tk.StringVar(value="—")

mean_est_var = tk.StringVar(value="—")
mean_theory_var = tk.StringVar(value="—")
mean_error_var = tk.StringVar(value="—")

dmu_est_var = tk.StringVar(value="—")
dmu_theory_var = tk.StringVar(value="—")
dmu_error_var = tk.StringVar(value="—")

dlambda_est_var = tk.StringVar(value="—")
dlambda_theory_var = tk.StringVar(value="—")
dlambda_error_var = tk.StringVar(value="—")

rows = [
    ("Mean system time estimate", mean_est_var, mean_theory_var, mean_error_var),
    ("IPA estimate dJ/dmu", dmu_est_var, dmu_theory_var, dmu_error_var),
    ("IPA estimate dJ/dlambda", dlambda_est_var, dlambda_theory_var, dlambda_error_var),
]

for i, row in enumerate(rows, start=2):
    ttk.Label(results_card, text=row[0], style="Body.TLabel").grid(row=i, column=0, padx=20, pady=4, sticky="w")
    ttk.Label(results_card, textvariable=row[1], style="Body.TLabel").grid(row=i, column=1, padx=20, pady=4, sticky="w")
    ttk.Label(results_card, textvariable=row[2], style="Body.TLabel").grid(row=i, column=2, padx=20, pady=4, sticky="w")
    ttk.Label(results_card, textvariable=row[3], style="Body.TLabel").grid(row=i, column=3, padx=20, pady=4, sticky="w")

# --------------------------------------------------
# Plot card
# --------------------------------------------------
plot_card = ttk.Frame(main_frame, style="Card.TFrame", padding=10)
plot_card.pack(fill="both", expand=True, padx=16, pady=(0, 16))

ttk.Label(plot_card, text="Plots", style="CardTitle.TLabel").pack(anchor="w", padx=6, pady=(4, 8))

plot_frame = ttk.Frame(plot_card, style="Card.TFrame")
plot_frame.pack(fill="both", expand=True)

root.mainloop()