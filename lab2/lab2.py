import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math

def poly_to_string(coeffs, var='t', precision=6):
    terms = []
    degree = len(coeffs) - 1
    for i in range(degree, -1, -1):
        coeff = coeffs[i]
        if abs(coeff) < 1e-12:
            continue

        sign = "+" if coeff > 0 else "-"
        abs_coeff = abs(coeff)

        coeff_str = f"{abs_coeff:.{precision}f}"

        if i == 0:
            term = coeff_str
        elif i == 1:
            if abs(abs_coeff - 1.0) < 1e-12:
                term = var
            else:
                term = f"{coeff_str}*{var}"
        else:
            if abs(abs_coeff - 1.0) < 1e-12:
                term = f"{var}^{i}"
            else:
                term = f"{coeff_str}*{var}^{i}"

        if not terms:
            if sign == "-":
                terms.append(f"-{term}")
            else:
                terms.append(term)
        else:
            terms.append(f" {sign} {term}")

    if not terms:
        return "0"
    else:
        return "".join(terms)

def poly_scalar_mul(coeffs, scalar):
    return [c * scalar for c in coeffs]

def poly_sub(coeffs1, coeffs2):
    len1 = len(coeffs1)
    len2 = len(coeffs2)
    max_len = max(len1, len2)
    padded1 = coeffs1 + [0.0] * (max_len - len1)
    padded2 = coeffs2 + [0.0] * (max_len - len2)
    return [c1 - c2 for c1, c2 in zip(padded1, padded2)]

a = 0.0
b = 1.7
weight_func = lambda t: (6 - t)**2
target_func = lambda t: np.cos(2 * t)
epsilons = [1e-1, 1e-2, 1e-3]

def inner_product(f, g):
    if not callable(f) or not callable(g):
        raise TypeError(f"Arguments to inner_product must be callable. Got: {type(f)}, {type(g)}")
    integrand = lambda t: f(t) * g(t) * weight_func(t)
    result, error = quad(integrand, a, b, limit=200, epsabs=1e-9, epsrel=1e-9)
    if error > 1e-5:
        print(f"Warning: High integration error ({error:.2e}) for integral. Result: {result:.4e}")
    return result

max_potential_N = 10
x_basis_funcs = [lambda t, n=i: t**(n-1) for i in range(1, max_potential_N + 1)]
x_basis_coeffs = [[0.0]*(i-1) + [1.0] for i in range(1, max_potential_N + 1)]

phi_ortho_funcs = []
phi_ortho_coeffs = []
phi_ortho_str = []
phi_norms_sq = []
proj_coeffs_memo = {}

print("Начинаем процесс ортогонализации Грама-Шмидта...")

phi_1_func = x_basis_funcs[0]
phi_1_coeffs = x_basis_coeffs[0]
norm_phi_1_sq = inner_product(phi_1_func, phi_1_func)

if norm_phi_1_sq < 1e-18:
    raise ValueError("Norm of phi_1 is too small.")

phi_ortho_funcs.append(phi_1_func)
phi_ortho_coeffs.append(phi_1_coeffs)
phi_norms_sq.append(norm_phi_1_sq)
phi_1_str = poly_to_string(phi_1_coeffs)
phi_ortho_str.append(phi_1_str)

print(f"phi_1(t) = {phi_1_str}")
print(f"||phi_1||^2 = {norm_phi_1_sq:.6f}")

for n in range(2, max_potential_N + 1):
    print(f"Calculating phi_{n}...")
    xn_func = x_basis_funcs[n-1]
    xn_coeffs = x_basis_coeffs[n-1]

    phi_n_coeffs_current = list(xn_coeffs)
    projections = []

    for k in range(n - 1):
        phi_k_func = phi_ortho_funcs[k]
        phi_k_coeffs = phi_ortho_coeffs[k]
        norm_phi_k_sq = phi_norms_sq[k]

        memo_key = (n, k + 1)
        if memo_key in proj_coeffs_memo:
             ip_xn_phik = proj_coeffs_memo[memo_key]
        else:
             ip_xn_phik = inner_product(xn_func, phi_k_func)
             proj_coeffs_memo[memo_key] = ip_xn_phik
        projections.append(ip_xn_phik)

        if abs(norm_phi_k_sq) < 1e-18:
             print(f"Warning: norm_sq[{k}] is near zero ({norm_phi_k_sq:.2e})")
             raise ValueError(f"Near-zero norm encountered for phi_{k+1} during phi_{n} coeff calculation.")

        proj_coeff = ip_xn_phik / norm_phi_k_sq
        term_to_subtract_coeffs = poly_scalar_mul(phi_k_coeffs, proj_coeff)
        phi_n_coeffs_current = poly_sub(phi_n_coeffs_current, term_to_subtract_coeffs)

    def create_phi_n(current_xn, phis_at_creation, norms_sq_at_creation, projs_at_creation):
        local_phis = list(phis_at_creation)
        local_norms_sq = list(norms_sq_at_creation)
        local_projs = list(projs_at_creation)
        num_k = len(local_phis)

        def phi_n_func(t):
            sum_term = 0.0
            for k_idx in range(num_k):
                 if abs(local_norms_sq[k_idx]) < 1e-18:
                      print(f"Warning: norm_sq[{k_idx}] is near zero ({local_norms_sq[k_idx]:.2e})")
                      raise ValueError(f"Near-zero norm encountered for phi_{k_idx+1} during phi_{n} calculation.")
                 coeff = local_projs[k_idx] / local_norms_sq[k_idx]
                 sum_term += coeff * local_phis[k_idx](t)
            return current_xn(t) - sum_term
        return phi_n_func

    phi_n_func = create_phi_n(xn_func, phi_ortho_funcs, phi_norms_sq, projections)
    norm_phi_n_sq = inner_product(phi_n_func, phi_n_func)

    if norm_phi_n_sq < 1e-15:
        print(f"Warning: Норма ||phi_{n}||^2 ({norm_phi_n_sq:.2e}) очень мала. Остановка ортогонализации.")
        max_potential_N = n - 1
        break

    phi_ortho_funcs.append(phi_n_func)
    phi_ortho_coeffs.append(phi_n_coeffs_current)
    phi_norms_sq.append(norm_phi_n_sq)
    phi_n_str = poly_to_string(phi_n_coeffs_current)
    phi_ortho_str.append(phi_n_str)

    if n <= 4:
        print(f"phi_{n}(t) = {phi_n_str}")

    print(f"||phi_{n}||^2 = {norm_phi_n_sq:.6f}")

print(f"Ортогонализация завершена. Получено {len(phi_ortho_funcs)} функций.")
calculated_N = len(phi_ortho_funcs)

print("\n--- Первые 4 ортогональные функции (s_n = phi_n) ---")
for i in range(min(4, calculated_N)):
    print(f"s{i+1}(t) = phi_{i+1}(t) = {phi_ortho_str[i]}")
if calculated_N < 4:
    print(f"Примечание: Рассчитано только {calculated_N} функций.")

norm_y_sq = inner_product(target_func, target_func)
print(f"\n||y||^2 = ||cos(2t)||^2 = {norm_y_sq:.6f}")

fourier_coeffs_c = []
fourier_coeffs_num = []
mse_history = []
N_values = {}

current_mse = norm_y_sq

print("\nВычисление коэффициентов Фурье и среднеквадратичной ошибки:")

for n in range(1, calculated_N + 1):
    phi_n = phi_ortho_funcs[n-1]
    norm_phi_n_sq = phi_norms_sq[n-1]

    if abs(norm_phi_n_sq) < 1e-18:
        print(f"Error: ||phi_{n}||^2 is near zero ({norm_phi_n_sq:.2e}) before calculating c_{n}. Stopping.")
        calculated_N = n - 1
        fourier_coeffs_c = fourier_coeffs_c[:n-1]
        fourier_coeffs_num = fourier_coeffs_num[:n-1]
        mse_history = mse_history[:n-1]
        if n > 1:
             current_mse = mse_history[-1]
        else:
             current_mse = norm_y_sq
        break

    ip_y_phin = inner_product(target_func, phi_n)
    cn = ip_y_phin / norm_phi_n_sq

    fourier_coeffs_c.append(cn)
    fourier_coeffs_num.append(ip_y_phin)

    term_n_mse = (ip_y_phin**2) / norm_phi_n_sq
    if term_n_mse < 0:
         print(f"Warning: MSE term for n={n} is negative ({term_n_mse:.2e}). Clamping to zero.")
         term_n_mse = 0

    current_mse -= term_n_mse
    current_mse = max(0, current_mse)
    mse_history.append(current_mse)

    print(f"N={n}: c_{n}={cn:.6f}, <y, phi_{n}>={ip_y_phin:.4f}, term_mse={term_n_mse:.4e}, Current MSE = {current_mse:.4e}")

    eps_to_check = list(eps for eps in epsilons if eps not in N_values)
    for eps in eps_to_check:
        if current_mse < eps:
            N_values[eps] = n
            print(f"  >> Достигнута точность MSE < {eps:.1e} при N = {n}")

for eps in sorted(epsilons):
    if eps not in N_values:
        print(f"Предупреждение: Не удалось достичь MSE < {eps:.1e} при N={calculated_N}. Финальное MSE = {current_mse:.3e}")

t_vals = np.linspace(a, b, 500)
y_vals = target_func(t_vals)

plt.figure(figsize=(12, 7))
num_phi_to_plot = min(5, calculated_N)
if num_phi_to_plot > 0:
    max_phi_val = 0
    min_phi_val = 0
    phi_vals_cache = {}
    for i in range(num_phi_to_plot):
        phi_i_func = phi_ortho_funcs[i]
        phi_i_vals = np.array([phi_i_func(t) for t in t_vals])
        phi_vals_cache[i] = phi_i_vals

        current_max = np.max(phi_i_vals)
        current_min = np.min(phi_i_vals)
        if i == 0:
             max_phi_val = current_max
             min_phi_val = current_min
        else:
             max_phi_val = max(max_phi_val, current_max)
             min_phi_val = min(min_phi_val, current_min)

        plt.plot(t_vals, phi_i_vals, label=f'$\\phi_{{{i+1}}}(t)$')

    plot_margin = (max_phi_val - min_phi_val) * 0.1 if max_phi_val > min_phi_val else 1.0
    plt.ylim(min_phi_val - plot_margin, max_phi_val + plot_margin)

    plt.title('Первые несколько ортогональных функций $\\phi_n(t)$')
    plt.xlabel('t')
    plt.ylabel('$\\phi_n(t)$')
    plt.legend()
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.show()
else:
    print("Недостаточно ортогональных функций для построения графика.")

plt.figure(figsize=(12, 8))
plt.plot(t_vals, y_vals, label='$y(t) = \\cos(2t)$', linewidth=2, color='black', linestyle=':')

def create_SN(coeffs, phis_funcs, current_N):
    local_coeffs = list(coeffs[:current_N])
    local_phis_funcs = list(phis_funcs[:current_N])
    def SN_func(t):
        s = 0.0
        for k in range(current_N):
            s += local_coeffs[k] * local_phis_funcs[k](t)
        return s
    return SN_func

num_intermediate_plots = min(4, calculated_N)
if num_intermediate_plots > 0 and len(fourier_coeffs_c) > 0:
    print(f"\nПостроение графика промежуточных сумм S_N(t) для N=1..{num_intermediate_plots}")
    for N_intermediate in range(1, num_intermediate_plots + 1):
        if N_intermediate <= len(fourier_coeffs_c) and N_intermediate <= len(phi_ortho_funcs):
            SN_func_intermediate = create_SN(fourier_coeffs_c, phi_ortho_funcs, N_intermediate)
            SN_vals_intermediate = np.array([SN_func_intermediate(t) for t in t_vals])
            plt.plot(t_vals, SN_vals_intermediate, label=f'$S_{{{N_intermediate}}}(t)$')
        else:
             print(f"Недостаточно данных для построения S_{N_intermediate}(t).")

    plt.title('Промежуточные приближения $S_N(t)$ для N=1..4')
    plt.xlabel('t')
    plt.ylabel('Значение функции')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Недостаточно данных для построения графика промежуточных сумм S_N(t).")

plt.figure(figsize=(12, 8))
plt.plot(t_vals, y_vals, label='$y(t) = \\cos(2t)$', linewidth=3, color='black', linestyle='--')

results = {}
sorted_eps = sorted(epsilons)
plotted_N = set()

found_N_values = sorted(list(set(N_values.values())))

if not found_N_values:
    print("Не найдено ни одного N, удовлетворяющего условиям точности. График приближений по эпсилон не строится.")
else:
    print("\nПостроение графика приближений S_N(t) для найденных N по эпсилон...")
    for N in found_N_values:
        if N > calculated_N:
            print(f"Предупреждение: Запрошен N={N}, но доступно только {calculated_N} функций. Пропуск.")
            continue
        if N <= 0: continue
        if N > len(fourier_coeffs_c) or N > len(phi_ortho_funcs):
             print(f"Предупреждение: Недостаточно данных (коэфф. или функций) для N={N}. Пропуск.")
             continue

        SN_func = create_SN(fourier_coeffs_c, phi_ortho_funcs, N)
        SN_vals = np.array([SN_func(t) for t in t_vals])

        corresponding_eps = [f"{eps:.0e}" for eps, n_val in N_values.items() if n_val == N]
        eps_label = "$\\epsilon \\leq$ " + ", ".join(corresponding_eps) if corresponding_eps else ""

        if N > 0 and N <= len(mse_history):
            actual_mse = mse_history[N-1]
            plt.plot(t_vals, SN_vals, label=f'$S_{{{N}}}(t)$ (N={N}, {eps_label}), MSE$\\approx${actual_mse:.2e}')
            plotted_N.add(N)
        else:
            print(f"Предупреждение: Невозможно получить MSE для N={N}. Пропуск.")


    plt.title('Приближение $y(t) = \\cos(2t)$ частичными суммами $S_N(t)$ ряда Фурье (по $\\epsilon$)')
    plt.xlabel('t')
    plt.ylabel('Значение функции')
    plt.legend()
    plt.grid(True)
    plt.show()

print("\n--- Итоги ---")
print(f"Интервал: [{a}, {b}]")
print(f"Весовая функция: f(t) = (6-t)^2")
print(f"Целевая функция: y(t) = cos(2t)")
print(f"Норма целевой функции: ||y||^2 = {norm_y_sq:.6f}")
print(f"Процесс остановлен на N = {calculated_N} из-за малой нормы или достижения max_potential_N.")
print("\nТребуемая точность и соответствующее число членов ряда N:")
print("-" * 40)
print("| Эпсилон (ε) |   N   | Фактич. MSE  |")
print("-" * 40)
for eps in sorted(epsilons):
    if eps in N_values:
        N = N_values[eps]
        if N > 0 and N <= len(mse_history):
            final_mse = mse_history[N-1]
            print(f"| {eps:<11.1e} | {N:^5} | {final_mse:<12.3e} |")
        else:
            print(f"| {eps:<11.1e} | {N:^5} | {'MSE N/A':<12} |")
    else:
        final_mse = mse_history[-1] if mse_history else norm_y_sq
        print(f"| {eps:<11.1e} | {' > ' + str(calculated_N):<5} | {final_mse:<12.3e} |")
print("-" * 40)

if mse_history:
    plt.figure(figsize=(10, 6))
    n_axis = np.arange(1, len(mse_history) + 1)
    plt.semilogy(n_axis, mse_history, marker='o', linestyle='-')
    for eps in epsilons:
        plt.axhline(eps, linestyle='--', color='r', label=f'$\\epsilon={eps:.0e}$')
    if N_values:
        unique_N_for_eps = sorted(list(set(N_values.values())))
        plotted_N_labels = set()
        for N_req in unique_N_for_eps:
             if N_req > 0 and N_req <= len(mse_history):
                 label = f'N={N_req}'
                 if label not in plotted_N_labels:
                     plt.vlines(N_req, plt.ylim()[0], mse_history[N_req-1], color='g', linestyle=':', label=label)
                     plotted_N_labels.add(label)

    plt.title('Среднеквадратичная ошибка $E_N = ||y - S_N||^2$')
    plt.xlabel('Число членов ряда (N)')
    plt.ylabel('MSE (логарифмическая шкала)')
    plt.grid(True, which="both")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(bottom=max(1e-9, min(mse_history)/10) if mse_history else 1e-9)
    plt.show()
else:
    print("Нет данных для построения графика MSE.")
