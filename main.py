from src import run_classical_random_walk, run_simulation, Params
import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# 1. КОНФИГУРАЦИЯ И ПАРАМЕТРЫ
# ==========================================

NODES = 10              # Количество узлов (0..9)
T_START: float = 0.0
T_HORIZON = 20.0        # Временной горизонт
TIME_SAMPLES: int = 100
GAMMA = 2.0             # Коэффициент диффузии для Рис. 2 и 3
ETA = 0.8               # Скорость считывания (sink rate) из узла 9
SEED = 42

# Фиксируем случайность для воспроизводимости потенциала V
np.random.seed(SEED)
# Потенциал V: линейный наклон + шум
# V_i = 0.3 * (9 - i + epsilon), где epsilon ~ U[-0.5, 0.5]
epsilon = np.random.uniform(-0.5, 0.5, NODES)
V_pot = 0.3 * (np.arange(NODES)[::-1] - 0 + epsilon)
# Сдвинем V, чтобы минимум был около 0 (для удобства, физика не меняется)
V_pot = V_pot - np.min(V_pot)

print("Potential profile generated based on Table 1.")

# ==========================================
# 3. ГЕНЕРАЦИЯ ГРАФИКОВ
# ==========================================


params = Params(NODES, T_START, T_HORIZON, TIME_SAMPLES, GAMMA, ETA, epsilon, V_pot)


print("Simulating Coherent dynamics (kappa=0)...")
p_coh, ev_coh = run_simulation(0.0, params)

print("Simulating QRN/Netting dynamics (kappa=1.0)...")
p_qrn, ev_qrn = run_simulation(1.0, params)

print("Simulating Classical Random Walk...")
p_crw, ev_crw = run_classical_random_walk(params)

# --- Figure 2: P_success(t) ---
plt.figure(figsize=(8, 5))
plt.plot(params.times, p_coh, label=r'Coherent ($\kappa=0$)', linestyle='--', color='blue')
plt.plot(params.times, p_qrn, label=r'QRN Netting ($\kappa=1$)', linewidth=3, color='green')
plt.plot(params.times, p_crw, label='Classical RW', linestyle=':', color='gray')
plt.title('Figure 2: Targeting Probability $P_{success}(t)$')
plt.xlabel('Time (a.u.)')
plt.ylabel('Cumulative Probability at Sink')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Figure_2_Targeting.png', dpi=300)
print("Figure 2 saved.")

# --- Figure 3: Expected Potential E[V](t) ---
plt.figure(figsize=(8, 5))
plt.plot(params.times, ev_coh, label=r'Coherent ($\kappa=0$)', linestyle='--', color='blue')
plt.plot(params.times, ev_qrn, label=r'QRN Netting ($\kappa=1$)', linewidth=3, color='green')
# Опционально добавим CRW, хотя в описании Fig 3 акцент на kappa=0 vs kappa=1
plt.plot(params.times, ev_crw, label='Classical RW', linestyle=':', color='gray', alpha=0.7)
plt.title('Figure 3: Expected Potential $E[V](t)$ (Free Energy Proxy)')
plt.xlabel('Time (a.u.)')
plt.ylabel('Expected Potential Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Figure_3_Potential.png', dpi=300)
print("Figure 3 saved.")

# --- Figure 4: Heatmap (ENAQT Bell Curve) ---
# "Optimum region appears at intermediate kappa"
print("Calculating Heatmap (this may take 10-20 seconds)...")

gammas = np.linspace(0.5, 4.0, 20)
kappas = np.linspace(0.0, 2.5, 20)
heatmap_data = np.zeros((len(kappas), len(gammas)))

# Для хитмапа нам нужна только конечная точка T=20
for i, k_val in enumerate(kappas):
    for j, g_val in enumerate(gammas):
        # Быстрый прогон
        # Можно оптимизировать, вычисляя только final state, но solve_ivp надежнее
        # Используем меньше точек t_eval для скорости
        temp_params = Params(params.nodes, params.t_start, params.t_horizon, 2, g_val, params.eta, params.epsilon, params.v_pot)
        res_p, _ = run_simulation(k_val, temp_params)
        heatmap_data[i, j] = res_p[-1]

plt.figure(figsize=(7, 6))
plt.imshow(heatmap_data, origin='lower', aspect='auto', cmap='viridis',
           extent=[gammas.min(), gammas.max(), kappas.min(), kappas.max()])
plt.colorbar(label='$P_{success}(T=20)$')
plt.title('Figure 4: ENAQT Region (Optimization landscape)')
plt.xlabel(r'Diffusion Strength $\gamma$')
plt.ylabel(r'Dephasing Rate $\kappa$')
plt.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='Slice for Figs 2-3')
plt.legend(loc='upper right')
plt.savefig('Figure_4_Heatmap.png', dpi=300)
print("Figure 4 saved.")

plt.show()
