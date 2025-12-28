import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]
# [NOT ACCESSED] from scipy.sparse import csr_matrix, kron, identity, eye

# ==========================================
# 1. КОНФИГУРАЦИЯ И ПАРАМЕТРЫ (из Таблицы 1)
# ==========================================

N = 10                  # Количество узлов (0..9)
T_HORIZON = 20.0        # Временной горизонт
GAMMA_DEFAULT = 2.0     # Коэффициент диффузии для Рис. 2 и 3
ETA = 0.8               # Скорость считывания (sink rate) из узла 9

# Фиксируем случайность для воспроизводимости потенциала V
np.random.seed(42)
# Потенциал V: линейный наклон + шум
# V_i = 0.3 * (9 - i + epsilon), где epsilon ~ U[-0.5, 0.5]
epsilon = np.random.uniform(-0.5, 0.5, N)
V_pot = 0.3 * (np.arange(N)[::-1] - 0 + epsilon)
# Сдвинем V, чтобы минимум был около 0 (для удобства, физика не меняется)
V_pot = V_pot - np.min(V_pot)

print("Potential profile generated based on Table 1.")

# ==========================================
# 2. ФИЗИЧЕСКИЙ ДВИЖОК (Линдблад)
# ==========================================

def get_hamiltonian(gamma, n_nodes, potential):
    """
    Создает Гамильтониан H_net = -gamma * L + V
    [span_1](start_span)
    """
    # Лапласиан графа (цепочка)
    # Диагональ: 2 (1 на краях), вне диагонали: -1
    # Но для цепочки в физике часто используют матрицу смежности: H ~ -gamma * A
    # В тексте сказано H = -gamma * L + V. Лапласиан L = D - A.
    # L для цепи:
    diagonals = np.zeros(n_nodes)
    off_diagonals = np.zeros(n_nodes - 1)
    
    for i in range(n_nodes):
        degree = 2 if 0 < i < n_nodes - 1 else 1
        diagonals[i] = degree
    
    # Строим матрицу Лапласиана
    L = np.diag(diagonals) - np.diag(np.ones(n_nodes-1), 1) - np.diag(np.ones(n_nodes-1), -1)
    
    # Потенциал V (диагональная матрица)
    V_mat = np.diag(potential)
    
    # H_net
    return -gamma * L + V_mat

def liouvillian(t, rho_vec, H, kappa, eta, n_nodes):
    """
    Супероператор для уравнения d rho / dt = -i[H, rho] + L_dephase + L_sink
    [span_1](end_span)
    Используем векторизацию (rho превращается в вектор n*n).
    """
    # Восстанавливаем матрицу плотности из вектора
    rho = rho_vec.reshape((n_nodes, n_nodes))
    
    # 1. Унитарная часть: -i [H, rho]
    comm = -1j * (H @ rho - rho @ H)
    
    # 2. Дефазировка (Dephasing): kappa * sum( Z_i rho Z_i - rho )
    # Эквивалентно подавлению недиагональных элементов со скоростью kappa
    # L_dephase[rho]_ij = -kappa * rho_ij (для i != j)
    # Диагональные элементы не меняются при чистой дефазировке
    dephase = np.zeros_like(rho, dtype=complex)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                dephase[i, j] = -kappa * rho[i, j]
                
    # 3. Сток (Sink/Readout) из последнего узла (индекс N-1)
    # Моделируем как потерю населенности из узла N-1 (Target)
    # d(rho)/dt = -eta/2 * { |s><s|, rho } ... упрощенно для вероятности успеха:
    # Мы просто добавляем анти-эрмитов член или явный оператор скачка.
    # Для расчета P_success = 1 - Tr(rho) (населенность, ушедшая в сток)
    # Оператор L_sink = sqrt(eta) * |sink><target| (внешний сток)
    # Вклад: -eta/2 * (|s><s| rho + rho |s><s|)
    
    target_idx = n_nodes - 1
    sink_term = np.zeros_like(rho, dtype=complex)
    
    # Затухание элемента target-target
    sink_term[target_idx, target_idx] -= eta * rho[target_idx, target_idx]
    
    # Затухание когерентностей, связанных с target (cross-terms)
    # off-diagonal decay rate is eta/2
    for k in range(n_nodes):
        if k != target_idx:
            sink_term[target_idx, k] -= (eta / 2.0) * rho[target_idx, k]
            sink_term[k, target_idx] -= (eta / 2.0) * rho[k, target_idx]

    d_rho = comm + dephase + sink_term
    return d_rho.flatten()

def run_simulation(gamma, kappa, eta, times):
    n_nodes = N
    H = get_hamiltonian(gamma, n_nodes, V_pot)
    
    # Начальное состояние: локализация в узле 0 (|0><0|)
    rho0 = np.zeros((n_nodes, n_nodes), dtype=complex)
    rho0[0, 0] = 1.0 + 0j
    rho0_vec = rho0.flatten()
    
    # Решаем ОДУ
    sol = solve_ivp(
        fun=lambda t, y: liouvillian(t, y, H, kappa, eta, n_nodes),
        t_span=(times[0], times[-1]),
        y0=rho0_vec,
        t_eval=times,
        method='RK45'
    )
    
    # Обработка результатов
    p_success = []
    expected_V = []
    
    for i in range(len(sol.t)):
        rho_t = sol.y[:, i].reshape((n_nodes, n_nodes))
        trace = np.real(np.trace(rho_t))
        
        # P_success - это то, что ушло в сток = 1 - Trace(rho)
        # (т.к. система открытая и мы теряем населенность в sink)
        p_success.append(1.0 - trace)
        
        # E[V] = Tr(rho * V) / Tr(rho) (нормализуем на оставшуюся вероятность)
        # Если Tr(rho) очень мал, считаем 0
        if trace > 1e-4:
            e_v = np.real(np.trace(rho_t @ np.diag(V_pot))) / trace
        else:
            e_v = 0 # или предыдущее значение
        expected_V.append(e_v)
        
    return np.array(p_success), np.array(expected_V)

def run_classical_random_walk(gamma, eta, times):
    """
    Классическое блуждание (CRW) для сравнения.
    Моделируем как Master Equation только для диагональных элементов (населенностей).
    Скорость перехода ~ gamma.
    """
    n_nodes = N
    # Матрица переходов K
    K = np.zeros((n_nodes, n_nodes))
    
    # Диффузия по цепочке
    for i in range(n_nodes):
        # Вправо
        if i < n_nodes - 1:
            K[i+1, i] += gamma
            K[i, i]   -= gamma
        # Влево
        if i > 0:
            K[i-1, i] += gamma
            K[i, i]   -= gamma
            
    # Сток из последнего узла
    K[n_nodes-1, n_nodes-1] -= eta
    
    # Начальное состояние
    p0 = np.zeros(n_nodes)
    p0[0] = 1.0
    
    def classical_deriv(t, p):
        return K @ p
        
    sol = solve_ivp(
        fun=classical_deriv,
        t_span=(times[0], times[-1]),
        y0=p0,
        t_eval=times
    )
    
    p_sink = []
    e_v = []
    for i in range(len(sol.t)):
        p_vec = sol.y[:, i]
        prob_remaining = np.sum(p_vec)
        p_sink.append(1.0 - prob_remaining)
        
        if prob_remaining > 1e-6:
            val = np.sum(p_vec * V_pot) / prob_remaining
        else:
            val = 0
        e_v.append(val)
        
    return np.array(p_sink), np.array(e_v)

# ==========================================
# 3. ГЕНЕРАЦИЯ ГРАФИКОВ
# ==========================================

times = np.linspace(0, T_HORIZON, 100)

print("Simulating Coherent dynamics (kappa=0)...")
p_coh, ev_coh = run_simulation(gamma=GAMMA_DEFAULT, kappa=0.0, eta=ETA, times=times)

print("Simulating QRN/Netting dynamics (kappa=1.0)...")
p_qrn, ev_qrn = run_simulation(gamma=GAMMA_DEFAULT, kappa=1.0, eta=ETA, times=times)

print("Simulating Classical Random Walk...")
p_crw, ev_crw = run_classical_random_walk(gamma=GAMMA_DEFAULT, eta=ETA, times=times)

# --- Figure 2: P_success(t) ---
plt.figure(figsize=(8, 5))
plt.plot(times, p_coh, label='Coherent ($\kappa=0$)', linestyle='--', color='blue')
plt.plot(times, p_qrn, label='QRN Netting ($\kappa=1$)', linewidth=3, color='green')
plt.plot(times, p_crw, label='Classical RW', linestyle=':', color='gray')
plt.title('Figure 2: Targeting Probability $P_{success}(t)$')
plt.xlabel('Time (a.u.)')
plt.ylabel('Cumulative Probability at Sink')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('Figure_2_Targeting.png', dpi=300)
print("Figure 2 saved.")

# --- Figure 3: Expected Potential E[V](t) ---
plt.figure(figsize=(8, 5))
plt.plot(times, ev_coh, label='Coherent ($\kappa=0$)', linestyle='--', color='blue')
plt.plot(times, ev_qrn, label='QRN Netting ($\kappa=1$)', linewidth=3, color='green')
# Опционально добавим CRW, хотя в описании Fig 3 акцент на kappa=0 vs kappa=1
plt.plot(times, ev_crw, label='Classical RW', linestyle=':', color='gray', alpha=0.7)
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
        res_p, _ = run_simulation(g_val, k_val, ETA, [0, T_HORIZON])
        heatmap_data[i, j] = res_p[-1]

plt.figure(figsize=(7, 6))
plt.imshow(heatmap_data, origin='lower', aspect='auto', cmap='viridis',
           extent=[gammas.min(), gammas.max(), kappas.min(), kappas.max()])
plt.colorbar(label='$P_{success}(T=20)$')
plt.title('Figure 4: ENAQT Region (Optimization landscape)')
plt.xlabel('Diffusion Strength $\gamma$')
plt.ylabel('Dephasing Rate $\kappa$')
plt.axhline(y=1.0, color='white', linestyle='--', alpha=0.5, label='Slice for Figs 2-3')
plt.legend(loc='upper right')
plt.savefig('Figure_4_Heatmap.png', dpi=300)
print("Figure 4 saved.")

plt.show()
