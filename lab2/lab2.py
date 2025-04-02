import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import math

# --- Шаг 1: Определения ---
a = 0.0
b = 1.7
weight_func = lambda t: (6 - t)**2
target_func = lambda t: np.cos(2 * t)
epsilons = [1e-1, 1e-2, 1e-3]

# Функция для вычисления скалярного произведения
def inner_product(f, g):
    if not callable(f) or not callable(g):
        raise TypeError(f"Arguments to inner_product must be callable. Got: {type(f)}, {type(g)}")
    integrand = lambda t: f(t) * g(t) * weight_func(t)
    result, error = quad(integrand, a, b, limit=200, epsabs=1e-9, epsrel=1e-9)
    if error > 1e-5: # Немного увеличим порог предупреждения
        print(f"Warning: High integration error ({error:.2e}) for integral. Result: {result:.4e}")
    return result

# --- Шаг 2: Ортогонализация Грама-Шмидта ---
max_potential_N = 10
x_basis = [lambda t, n=i: t**(n-1) for i in range(1, max_potential_N + 1)]

phi_ortho = []
phi_norms_sq = []
proj_coeffs_memo = {}

print("Начинаем процесс ортогонализации Грама-Шмидта...")

# Первый элемент
phi_1 = x_basis[0]
norm_phi_1_sq = inner_product(phi_1, phi_1)
if norm_phi_1_sq < 1e-18:
    raise ValueError("Norm of phi_1 is too small.")
phi_ortho.append(phi_1)
phi_norms_sq.append(norm_phi_1_sq)
print(f"phi_1(t) = 1")
print(f"||phi_1||^2 = {norm_phi_1_sq:.6f}")

# Последующие элементы
for n in range(2, max_potential_N + 1):
    print(f"Calculating phi_{n}...")
    xn = x_basis[n-1]

    projections = []
    for k in range(n - 1): # k от 0 до n-2 (индексы phi_1 ... phi_{n-1})
        phi_k = phi_ortho[k]
        # Ключ для memoization: (индекс x_n, индекс phi_k)
        memo_key = (n, k + 1)
        if memo_key in proj_coeffs_memo:
             ip_xn_phik = proj_coeffs_memo[memo_key]
             #print(f"  Using memoized <x_{n}, phi_{k+1}>")
        else:
             #print(f"  Calculating <x_{n}, phi_{k+1}>...")
             ip_xn_phik = inner_product(xn, phi_k)
             proj_coeffs_memo[memo_key] = ip_xn_phik
        projections.append(ip_xn_phik)
        #print(f"  <x_{n}, phi_{k+1}> = {ip_xn_phik:.6f}")

    # Создаем функцию для phi_n, которая правильно захватывает переменные
    # Передаем копии списков и используем их внутри
    def create_phi_n(current_xn, phis_at_creation, norms_sq_at_creation, projs_at_creation):
        local_phis = list(phis_at_creation)
        local_norms_sq = list(norms_sq_at_creation)
        local_projs = list(projs_at_creation)
        num_k = len(local_phis)

        def phi_n_func(t):
            sum_term = 0.0
            for k_idx in range(num_k):
                 # Проверка деления на ноль/малое число
                 if abs(local_norms_sq[k_idx]) < 1e-18:
                      print(f"Warning: norm_sq[{k_idx}] is near zero ({local_norms_sq[k_idx]:.2e})")
                      raise ValueError(f"Near-zero norm encountered for phi_{k_idx+1} during phi_{n} calculation.")

                 coeff = local_projs[k_idx] / local_norms_sq[k_idx]
                 sum_term += coeff * local_phis[k_idx](t)
            return current_xn(t) - sum_term
        return phi_n_func

    # Передаем текущее состояние списков в фабричную функцию
    phi_n_func = create_phi_n(xn, phi_ortho, phi_norms_sq, projections)
    # ------------------------

    # Вычисляем норму новой функции
    norm_phi_n_sq = inner_product(phi_n_func, phi_n_func)

    # Проверка на численную неустойчивость
    if norm_phi_n_sq < 1e-15:
        print(f"Warning: Норма ||phi_{n}||^2 ({norm_phi_n_sq:.2e}) очень мала. Остановка ортогонализации.")
        max_potential_N = n - 1
        break

    phi_ortho.append(phi_n_func)
    phi_norms_sq.append(norm_phi_n_sq)
    print(f"||phi_{n}||^2 = {norm_phi_n_sq:.6f}")


print(f"Ортогонализация завершена. Получено {len(phi_ortho)} функций.")
calculated_N = len(phi_ortho)

# --- Шаг 3 и 4: Коэффициенты Фурье и Определение N ---
# Вычисляем квадрат нормы целевой функции
norm_y_sq = inner_product(target_func, target_func)
print(f"\n||y||^2 = ||cos(2t)||^2 = {norm_y_sq:.6f}")

fourier_coeffs_c = []
fourier_coeffs_num = []
mse_history = []
N_values = {}

current_mse = norm_y_sq

print("\nВычисление коэффициентов Фурье и среднеквадратичной ошибки:")

# Убедимся, что итерируемся до фактически посчитанного числа функций
for n in range(1, calculated_N + 1):
    phi_n = phi_ortho[n-1]
    norm_phi_n_sq = phi_norms_sq[n-1]

    # Проверим норму еще раз перед делением
    if abs(norm_phi_n_sq) < 1e-18:
        print(f"Error: ||phi_{n}||^2 is near zero ({norm_phi_n_sq:.2e}) before calculating c_{n}. Stopping.")
        # Обрежем результаты до n-1
        calculated_N = n - 1
        fourier_coeffs_c = fourier_coeffs_c[:n-1]
        fourier_coeffs_num = fourier_coeffs_num[:n-1]
        mse_history = mse_history[:n-1]
        if n > 1:
             current_mse = mse_history[-1]
        else:
             current_mse = norm_y_sq
        break

    # Коэффициент Фурье
    ip_y_phin = inner_product(target_func, phi_n)
    cn = ip_y_phin / norm_phi_n_sq

    fourier_coeffs_c.append(cn)
    fourier_coeffs_num.append(ip_y_phin)

    # Обновление MSE
    term_n_mse = (ip_y_phin**2) / norm_phi_n_sq
    # Добавим проверку на случай отрицательного term_n_mse из-за ошибок
    if term_n_mse < 0:
         print(f"Warning: MSE term for n={n} is negative ({term_n_mse:.2e}). Clamping to zero.")
         term_n_mse = 0

    current_mse -= term_n_mse
    current_mse = max(0, current_mse)
    mse_history.append(current_mse)

    print(f"N={n}: c_{n}={cn:.6f}, <y, phi_{n}>={ip_y_phin:.4f}, term_mse={term_n_mse:.4e}, Current MSE = {current_mse:.4e}")

    # Проверка условия для epsilon
    # Создаем копию, чтобы безопасно удалять найденные
    eps_to_check = list(eps for eps in epsilons if eps not in N_values)
    for eps in eps_to_check:
        if current_mse < eps:
            N_values[eps] = n
            print(f"  >> Достигнута точность MSE < {eps:.1e} при N = {n}")

# Проверка, найдены ли N для всех epsilon
for eps in sorted(epsilons):
    if eps not in N_values:
        print(f"Предупреждение: Не удалось достичь MSE < {eps:.1e} при N={calculated_N}. Финальное MSE = {current_mse:.3e}")

# --- Шаг 5: Построение графиков ---
# (Этот блок тоже без изменений, но он будет использовать исправленные phi_ortho и fourier_coeffs_c)

t_vals = np.linspace(a, b, 500)
y_vals = target_func(t_vals)

# Графики первых нескольких ортогональных функций
plt.figure(figsize=(12, 7))
num_phi_to_plot = min(5, calculated_N)
if num_phi_to_plot > 0:
    max_phi_val = 0
    min_phi_val = 0
    for i in range(num_phi_to_plot):
        phi_i_vals = np.array([phi_ortho[i](t) for t in t_vals])
        max_phi_val = max(max_phi_val, np.max(phi_i_vals))
        min_phi_val = min(min_phi_val, np.min(phi_i_vals))
        plt.plot(t_vals, phi_i_vals, label=f'$\\phi_{i+1}(t)$')

    # Установка пределов оси Y на основе фактических значений
    plot_margin = (max_phi_val - min_phi_val) * 0.1
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


# Графики приближений S_N(t)
plt.figure(figsize=(12, 8))
plt.plot(t_vals, y_vals, label='$y(t) = \cos(2t)$', linewidth=3, color='black', linestyle='--')

results = {}
sorted_eps = sorted(epsilons)
plotted_N = set()

# Строим графики для N, которые были найдены
found_N_values = sorted(list(set(N_values.values())))

if not found_N_values:
     print("Не найдено ни одного N, удовлетворяющего условиям точности. График приближений не строится.")
else:
    for N in found_N_values:
        if N > calculated_N: # Убедимся, что N не превышает кол-во расчитанных phi
            print(f"Предупреждение: Запрошен N={N}, но доступно только {calculated_N} функций. Пропуск.")
            continue
        if N <= 0: continue # Пропуск невалидных N

        # Конструируем частичную сумму S_N(t)
        def create_SN(coeffs, phis, current_N):
            # Захватываем нужные части списков
            local_coeffs = list(coeffs[:current_N])
            local_phis = list(phis[:current_N])
            def SN_func(t):
                s = 0.0
                for k in range(current_N):
                    s += local_coeffs[k] * local_phis[k](t)
                return s
            return SN_func

        # Используем только рассчитанные коэффициенты и функции
        SN_func = create_SN(fourier_coeffs_c, phi_ortho, N)
        SN_vals = np.array([SN_func(t) for t in t_vals])

        # Найдем эпсилон(ы), которому соответствует данный N
        corresponding_eps = [f"{eps:.0e}" for eps, n_val in N_values.items() if n_val == N]
        eps_label = "$\\epsilon \\leq$ " + ", ".join(corresponding_eps) if corresponding_eps else ""

        actual_mse = mse_history[N-1] # Берем MSE из истории

        plt.plot(t_vals, SN_vals, label=f'$S_{N}(t)$ (N={N}, {eps_label}), MSE$\\approx${actual_mse:.2e}')
        plotted_N.add(N)

    plt.title('Приближение $y(t) = \cos(2t)$ частичными суммами $S_N(t)$ ряда Фурье')
    plt.xlabel('t')
    plt.ylabel('Значение функции')
    plt.legend()
    plt.grid(True)
    plt.show()

# Итоговая таблица результатов
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
        # Убедимся, что индекс N-1 существует в mse_history
        if N > 0 and N <= len(mse_history):
             final_mse = mse_history[N-1]
             print(f"| {eps:<11.1e} | {N:^5} | {final_mse:<12.3e} |")
        else:
             print(f"| {eps:<11.1e} | {N:^5} | {'MSE N/A':<12} |") # Если N=0 или ошибка
    else:
        # Если N не найден, показываем финальное MSE
        final_mse = mse_history[-1] if mse_history else norm_y_sq
        print(f"| {eps:<11.1e} | {' > ' + str(calculated_N):<5} | {final_mse:<12.3e} |")
print("-" * 40)

# График убывания ошибки MSE
if mse_history: # Строим график только если есть данные
    plt.figure(figsize=(10, 6))
    n_axis = np.arange(1, len(mse_history) + 1)
    plt.semilogy(n_axis, mse_history, marker='o', linestyle='-')
    # Добавим горизонтальные линии для уровней epsilon
    for eps in epsilons:
        plt.axhline(eps, linestyle='--', color='r', label=f'$\\epsilon = {eps:.0e}$')
        if eps in N_values:
             N_req = N_values[eps]
             # Убедимся, что индекс N_req-1 существует
             if N_req > 0 and N_req <= len(mse_history):
                  plt.vlines(N_req, plt.ylim()[0], mse_history[N_req-1], color='g', linestyle=':', label=f'N={N_req} for $\\epsilon={eps:.0e}$')

    plt.title('Среднеквадратичная ошибка $E_N = ||y - S_N||^2$')
    plt.xlabel('Число членов ряда (N)')
    plt.ylabel('MSE (логарифмическая шкала)')
    plt.grid(True, which="both")
    # Упорядочить легенду
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.ylim(bottom=max(1e-9, min(mse_history)/10)) # Установим нижний предел для лог. шкалы
    plt.show()
else:
    print("Нет данных для построения графика MSE.")
