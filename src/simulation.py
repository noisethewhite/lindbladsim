from .params import Params
from .types import FloatArray
from .physics import hamiltonian, solve_ivp, liouvillian
import numpy as np


def run_simulation(kappa: float, params: Params) -> tuple[FloatArray, FloatArray]:

    H = hamiltonian(params.gamma, params.nodes, params.v_pot)

    # Начальное состояние: локализация в узле 0 (|0><0|)
    rho0 = np.zeros((params.nodes, params.nodes), dtype=complex)
    rho0[0, 0] = 1.0 + 0j
    rho0_vec = rho0.flatten()

    # Решаем ОДУ
    sol = solve_ivp(
        fun=lambda t, y: liouvillian(t, y, H, kappa, params),
        t_span=(params.times[0], params.times[-1]),
        y0=rho0_vec,
        t_eval=params.times,
        method='RK45'
    )

    # Обработка результатов
    p_success: list[np.floating] = []
    expected_V: list[np.floating] = []

    for i in range(len(sol.t)):
        rho_t = sol.y[:, i].reshape((params.nodes, params.nodes))
        trace = np.real(np.trace(rho_t))

        # P_success - это то, что ушло в сток = 1 - Trace(rho)
        # (т.к. система открытая и мы теряем населенность в sink)
        p_success.append(1.0 - trace)

        # E[V] = Tr(rho * V) / Tr(rho) (нормализуем на оставшуюся вероятность)
        # Если Tr(rho) очень мал, считаем 0
        if trace > 1e-4:
            e_v = np.real(np.trace(rho_t @ np.diag(params.v_pot))) / trace
        else:
            e_v = np.float64(0) # или предыдущее значение
        expected_V.append(e_v)

    return np.array(p_success), np.array(expected_V)
