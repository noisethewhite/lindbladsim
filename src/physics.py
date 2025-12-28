from .params import Params
from .types import F64Array, FloatArray, TimeVector, FloatMatrix, ComplexArray, ComplexVector
from typing import Callable, Optional, Sequence, \
    Tuple, Union, TypeVar, Any, Protocol
import numpy as np
from scipy.integrate import solve_ivp as solve_ivp_untyped # pyright: ignore[reportMissingTypeStubs, reportUnknownVariableType]


_TF64Array = TypeVar("_TF64Array", bound=F64Array)


class OdeSolution(Protocol):
    t: TimeVector
    y: FloatMatrix


def solve_ivp(
    fun: Callable[
        [float, _TF64Array],
        Union[_TF64Array, Sequence[float]]
    ],
    t_span: Tuple[float, float],
    y0: _TF64Array,
    *,
    method: str = 'RK45',
    t_eval: Optional[Sequence[float] | F64Array] = None,
    dense_output: bool = False,
    events: Optional[Union[
        Callable[[float, _TF64Array], float],
        Sequence[Callable[[float, _TF64Array], float]]]
    ] = None,
    vectorized: bool = False,
    args: Optional[Tuple[Any, ...]] = None,
    **options: object
) -> OdeSolution:
    return solve_ivp_untyped(
        fun, t_span, y0, method, t_eval,
        dense_output, events, vectorized, args, **options
    ) # pyright: ignore[reportUnknownVariableType]


def hamiltonian(gamma: float, n_nodes: int, potential: F64Array) -> F64Array:
    """
    Создаёт Гамильтониан H_net = -gamma * L + V
    """
    # Лапласиан графа (цепочка)
    # Диагональ: 2 (1 на краях), вне диагонали: -1
    # Но для цепочки в физике часто используют матрицу смежности: H ~ -gamma * A
    # В тексте сказано H = -gamma * L + V. Лапласиан L = D - A.
    # L для цепи:
    diagonals = np.zeros(n_nodes)

    for i in range(n_nodes):
        degree = 2 if 0 < i < n_nodes - 1 else 1
        diagonals[i] = degree

    # Строим матрицу Лапласиана
    L = np.diag(diagonals) - np.diag(np.ones(n_nodes-1), 1) - np.diag(np.ones(n_nodes-1), -1)

    # Потенциал V (диагональная матрица)
    V_mat = np.diag(potential)

    # H_net
    return -gamma * L + V_mat


def liouvillian(_: float, rho_vec: F64Array, H: F64Array, kappa: float, params: Params) -> ComplexVector:
    """
    Супероператор для уравнения d rho / dt = -i[H, rho] + L_dephase + L_sink
    Используем векторизацию (rho превращается в вектор n*n).
    """
    # Восстанавливаем матрицу плотности из вектора
    rho: FloatMatrix = rho_vec.reshape((params.nodes, params.nodes))

    # 1. Унитарная часть: -i [H, rho]
    comm: ComplexArray = -1j * (H @ rho - rho @ H)

    # 2. Дефазировка (Dephasing): kappa * sum( Z_i rho Z_i - rho )
    # Эквивалентно подавлению недиагональных элементов со скоростью kappa
    # L_dephase[rho]_ij = -kappa * rho_ij (для i != j)
    # Диагональные элементы не меняются при чистой дефазировке
    dephase: FloatMatrix = np.zeros_like(rho, dtype=complex)
    for i in range(params.nodes):
        for j in range(params.nodes):
            if i != j:
                dephase[i, j] = -kappa * rho[i, j]

    # 3. Сток (Sink/Readout) из последнего узла (индекс N-1)
    # Моделируем как потерю населенности из узла N-1 (Target)
    # d(rho)/dt = -eta/2 * { |s><s|, rho } ... упрощенно для вероятности успеха:
    # Мы просто добавляем анти-эрмитов член или явный оператор скачка.
    # Для расчета P_success = 1 - Tr(rho) (населенность, ушедшая в сток)
    # Оператор L_sink = sqrt(eta) * |sink><target| (внешний сток)
    # Вклад: -eta/2 * (|s><s| rho + rho |s><s|)

    target_idx = params.nodes - 1
    sink_term: FloatMatrix = np.zeros_like(rho, dtype=complex)

    # Затухание элемента target-target
    sink_term[target_idx, target_idx] -= params.eta * rho[target_idx, target_idx]

    # Затухание когерентностей, связанных с target (cross-terms)
    # off-diagonal decay rate is eta/2
    for k in range(params.nodes):
        if k != target_idx:
            sink_term[target_idx, k] -= (params.eta / 2.0) * rho[target_idx, k]
            sink_term[k, target_idx] -= (params.eta / 2.0) * rho[k, target_idx]

    d_rho = comm + dephase + sink_term
    return d_rho.flatten()


def run_classical_random_walk(params: Params) -> tuple[FloatArray, F64Array]:
    """
    Классическое блуждание (CRW) для сравнения.
    Моделируем как Master Equation только для диагональных элементов (населенностей).
    Скорость перехода ~ gamma.
    """

    # Матрица переходов K
    K = np.zeros((params.nodes, params.nodes))

    # Диффузия по цепочке
    for i in range(params.nodes):
        # Вправо
        if i < params.nodes - 1:
            K[i+1, i] += params.gamma
            K[i, i]   -= params.gamma
        # Влево
        if i > 0:
            K[i-1, i] += params.gamma
            K[i, i]   -= params.gamma

    # Сток из последнего узла
    K[params.nodes-1, params.nodes-1] -= params.eta

    # Начальное состояние
    p0 = np.zeros(params.nodes)
    p0[0] = 1.0

    sol = solve_ivp(
        fun=lambda _, p: K @ p,
        t_span=(params.times[0], params.times[-1]),
        y0=p0,
        t_eval=params.times
    )

    p_sink: list[np.floating] = []
    e_v: list[np.float64] = []
    for i in range(len(sol.t)):
        p_vec = sol.y[:, i]
        prob_remaining = np.sum(p_vec)
        p_sink.append(1.0 - prob_remaining)
        if prob_remaining > 1e-6:
            val = np.sum(p_vec * params.v_pot) / prob_remaining
        else:
            val = np.float64(0)
        e_v.append(val)
    return np.array(p_sink), np.array(e_v)
