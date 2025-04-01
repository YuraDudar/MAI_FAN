import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import warnings

print(f"--- Лабораторная работа 2/Выполнил Юрий Дударь ---")
print(f"Вариант: 6, k=9, l=15")
# 1. Определение констант и функций
a = 0.0
b = 1.7
weight_func = lambda t: (6 - t)**2
target_func = lambda t: np.cos(2 * t)

# Исходная система функций x_n(t) = t^(n-1)
max_potential_N = 15 # Макс. кол-во базисных функций для генерации
initial_funcs_sympy = []
initial_funcs = []
for n in range(1, max_potential_N + 1):
    func = lambda t, p=n-1: t**p
    initial_funcs.append(func)


# 2. Определение скалярного произведения и нормы
def inner_product(f, g):
    integrand = lambda t: f(t) * g(t) * weight_func(t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", spi.IntegrationWarning)
        result, err = spi.quad(integrand, a, b, limit=200)
    # Проверим ошибку интегрирования
    if err > 1e-6:
        print(f"Предупреждение: Возможна высокая погрешность интегрирования: {err:.2e}")
    return result

def norm_sq(f):
    return inner_product(f, f)

def norm(f):
    return np.sqrt(norm_sq(f))

# 3. Процесс Грама-Шмидта
ortho_funcs_unnormalized = [] # Список для phi_k^*
ortho_funcs_normalized = []   # Список для phi_k
norms_sq_list = []            # Список для ||phi_k^*||^2
max_N_generated = 0

print("Начало процесса Грама-Шмидта...")
for n in range(len(initial_funcs)):
    xn = initial_funcs[n]
    phi_star_n = xn

    # Вычитаем проекции на предыдущие ортогональные функции (phi_k^*)
    for k in range(n):
        phi_star_k = ortho_funcs_unnormalized[k]
        ip_xn_phistark = inner_product(xn, phi_star_k)
        norm_sq_phistark = norms_sq_list[k]

        # Коэффициент проекции
        projection_coeff = ip_xn_phistark / norm_sq_phistark

        # Обновляем phi_star_n вычитанием проекции
        phi_star_n = lambda t, current_f=phi_star_n, coeff=projection_coeff, prev_phi=phi_star_k: \
                     current_f(t) - coeff * prev_phi(t)

    # Вычисляем квадрат нормы полученной ортогональной функции phi_star_n
    current_norm_sq = norm_sq(phi_star_n)

    # Проверка на численную стабильность / линейную зависимость
    if current_norm_sq < 1e-18: # Порог малости нормы
        print(f"Внимание: Квадрат нормы на шаге {n+1} близок к нулю ({current_norm_sq:.2e}).")
        print("Возможна линейная зависимость или численные проблемы. Ортогонализация прекращена.")
        max_N_generated = n
        break

    # Нормализуем
    current_norm = np.sqrt(current_norm_sq)
    phi_n = lambda t, f=phi_star_n, norm_val=current_norm: f(t) / norm_val

    # Сохраняем результаты
    ortho_funcs_unnormalized.append(phi_star_n)
    ortho_funcs_normalized.append(phi_n)
    norms_sq_list.append(current_norm_sq)
    max_N_generated = n + 1
    print(f"Сгенерирована ортонормированная функция phi_{n+1}(t). ||phi_{n+1}^*||^2 = {current_norm_sq:.4e}")

print(f"\nПроцесс Грама-Шмидта завершен. Сгенерировано {max_N_generated} ортонормированных базисных функций.")

# 4. Вычисление коэффициентов Фурье для y(t) = cos(2t)
fourier_coeffs = []
print("\nВычисление коэффициентов Фурье c_k = <y, phi_k>...")
for k in range(max_N_generated):
    phi_k = ortho_funcs_normalized[k]
    ck = inner_product(target_func, phi_k)
    fourier_coeffs.append(ck)
    print(f"c_{k+1} = {ck:.6f}")

# 5. Расчет ошибки и определение N для каждого epsilon
norm_y_sq = norm_sq(target_func)
print(f"\nКвадрат нормы целевой функции ||y||^2 = {norm_y_sq:.6f}")

errors = []
sum_ck_sq = 0
required_N = {}
epsilons = [1e-1, 1e-2, 1e-3]

print("\nРасчет среднеквадратичной ошибки E_N = ||y||^2 - sum(c_k^2)...")
achieved_eps = {eps: False for eps in epsilons}

for N in range(1, max_N_generated + 1):
    sum_ck_sq += fourier_coeffs[N-1]**2
    error_N = norm_y_sq - sum_ck_sq
    # Ошибка не может быть отрицательной из-за численных погрешностей
    error_N = max(0, error_N)
    errors.append(error_N)
    print(f"N = {N}, sum(c_k^2) = {sum_ck_sq:.6f}, E_N = {error_N:.6e}")

    # Проверка достижения требуемой точности
    for eps in epsilons:
        if not achieved_eps[eps] and error_N < eps:
             required_N[eps] = N
             achieved_eps[eps] = True
             print(f"----> Достигнута точность epsilon = {eps:.0e} при N = {N}")

    if all(achieved_eps.values()):
        break

print("\nРезультаты определения N:")
for eps in epsilons:
    if eps in required_N:
        print(f"Для epsilon = {eps:.0e} требуется N = {required_N[eps]}")
    else:
        print(f"Точность epsilon = {eps:.0e} не достигнута с {max_N_generated} функциями. Последняя ошибка: {errors[-1]:.6e}")
        required_N[eps] = -1


# 6. Определение функции частичной суммы ряда Фурье
def get_partial_sum_func(N_terms):
    if N_terms <= 0 or N_terms > len(fourier_coeffs):
        print(f"Предупреждение: некорректное N={N_terms}. Возвращена нулевая функция.")
        return lambda t: 0

    coeffs_N = fourier_coeffs[:N_terms]
    basis_funcs_N = ortho_funcs_normalized[:N_terms]

    # Создаем функцию суммы
    def S_N(t):
        total = 0.0
        t_arr = np.asarray(t)
        is_scalar_input = t_arr.ndim == 0
        t_arr = t_arr.reshape(-1)

        result_arr = np.zeros_like(t_arr, dtype=float)
        for k in range(N_terms):
            phi_k_values = basis_funcs_N[k](t_arr)
            result_arr += coeffs_N[k] * phi_k_values

        # Возвращаем скаляр, если на входе был скаляр
        return result_arr.item() if is_scalar_input else result_arr

    return S_N

# 7. Генерация Графиков
t_values = np.linspace(a, b, 500)
y_values = target_func(t_values)

# График 1: y(t) и итоговые приближения S_N(t) для каждого epsilon
plt.figure(figsize=(12, 7))
plt.plot(t_values, y_values, label='$y(t) = \cos(2t)$', linewidth=2.5, color='black')

plotted_N = set()
for eps in sorted(epsilons):
    N = required_N.get(eps, -1)
    if N > 0 and N not in plotted_N:
        SN_func = get_partial_sum_func(N)
        SN_values = SN_func(t_values)
        plt.plot(t_values, SN_values, label=f'$S_{N}(t)$ ($\epsilon \leq {eps:.0e}$)', linestyle='--', linewidth=1.5)
        plotted_N.add(N)

plt.title('Функция $y(t) = \cos(2t)$ и её приближения $S_N(t)$')
plt.xlabel('$t$')
plt.ylabel('Значение функции')
plt.legend()
plt.grid(True)
plt.ylim(y_values.min() - 0.2, y_values.max() + 0.2)
plt.show()


# График 2: Сходимость среднеквадратичной ошибки
plt.figure(figsize=(10, 6))
N_range = range(1, len(errors) + 1)
plt.semilogy(N_range, errors, marker='o', linestyle='-', label='$E_N = ||y - S_N||^2$') # Логарифмическая шкала для ошибки

# Горизонтальные линии для epsilon
colors = ['red', 'green', 'blue']
for i, eps in enumerate(epsilons):
    plt.axhline(y=eps, color=colors[i], linestyle='--', label=f'$\epsilon = {eps:.0e}$')
    N_for_eps = required_N.get(eps, -1)
    # Вертикальная линия, где достигается точность
    if N_for_eps > 0:
         plt.axvline(x=N_for_eps, color=colors[i], linestyle=':',
                     label=f'N={N_for_eps} для $\epsilon={eps:.0e}$')


plt.title('Среднеквадратичная ошибка приближения $E_N$')
plt.xlabel('Число членов ряда $N$')
plt.ylabel('Ошибка $E_N$ (лог. шкала)')
plt.legend(loc='upper right')
plt.xticks(N_range)
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.show()


# График 3: Первые несколько ортонормированных базисных функций phi_k(t)
plt.figure(figsize=(12, 7))
num_basis_to_plot = min(max_N_generated, 4)
for k in range(num_basis_to_plot):
    phi_k_func = ortho_funcs_normalized[k]
    phi_k_values = phi_k_func(t_values)
    if np.max(np.abs(phi_k_values)) < 100:
      plt.plot(t_values, phi_k_values, label=f'$\\phi_{k+1}(t)$')
    else:
      print(f"График phi_{k+1}(t) пропущен из-за большого диапазона значений.")


plt.title('Первые несколько ортонормированных базисных функций $\phi_k(t)$')
plt.xlabel('$t$')
plt.ylabel('Значение функции $\phi_k(t)$')
plt.legend()
plt.grid(True)
plt.axhline(0, color='black', linewidth=0.5)
plt.show()

# График 4: Промежуточные приближения S_N(t)
plt.figure(figsize=(12, 7))
plt.plot(t_values, y_values, label='$y(t) = \cos(2t)$', linewidth=2.5, color='black')

intermediate_N_to_plot = []
last_N = 0
for eps in sorted(epsilons):
    N = required_N.get(eps, -1)
    if N > 0 and N != last_N:
        intermediate_N_to_plot.append(N)
        last_N = N
if max_N_generated <= 6:
    intermediate_N_to_plot = list(range(1, max_N_generated + 1))
else:
    intermediate_N_to_plot = sorted(list(set([1, 2, 3] + intermediate_N_to_plot)))

for N_inter in intermediate_N_to_plot:
     if N_inter > max_N_generated: continue
     SN_func = get_partial_sum_func(N_inter)
     SN_values = SN_func(t_values)
     plt.plot(t_values, SN_values, label=f'$S_{N_inter}(t)$', linestyle=':', linewidth=1.2)


plt.title('Промежуточные приближения $S_N(t)$ функции $y(t) = \cos(2t)$')
plt.xlabel('$t$')
plt.ylabel('Значение функции')
plt.legend(loc='lower left')
plt.grid(True)
plt.ylim(y_values.min() - 0.2, y_values.max() + 0.2)
plt.show()


# График 5: Ряд Фурье (частичные суммы S_N(t))
plt.figure(figsize=(12, 7))
plt.plot(t_values, y_values, label='$y(t) = \cos(2t)$', linewidth=2.5, color='black')

intermediate_N_to_plot = []
last_N = 0
# Включаем N, соответствующие требуемым epsilon
for eps in sorted(epsilons):
    N = required_N.get(eps, -1)
    if N > 0 and N != last_N:
        intermediate_N_to_plot.append(N)
        last_N = N
intermediate_N_to_plot = sorted(list(set([1, 2, 3] + intermediate_N_to_plot)))
intermediate_N_to_plot = [n for n in intermediate_N_to_plot if n <= max_N_generated] # Убираем N > max

# Строим графики для выбранных N
for N_inter in intermediate_N_to_plot:
     SN_func = get_partial_sum_func(N_inter)
     SN_values = SN_func(t_values)
     plt.plot(t_values, SN_values, label=f'$S_{N_inter}(t)$', linestyle='--', linewidth=1.2)

plt.title('График частичных сумм ряда Фурье $S_N(t)$ в сравнении с $y(t)$')
plt.xlabel('$t$')
plt.ylabel('Значение функции')
plt.legend(loc='lower left')
plt.grid(True)
plt.ylim(y_values.min() - 0.2, y_values.max() + 0.2)
plt.show()

# График 6: Коэффициенты Фурье c_k
plt.figure(figsize=(10, 6))
k_values = np.arange(1, max_N_generated + 1)
plt.stem(k_values, fourier_coeffs[:max_N_generated], basefmt=" ", linefmt='b-', markerfmt='bo')
# Добавим логарифмическую шкалу по y для наглядности убывания
plt.yscale('symlog', linthresh=1e-5)
plt.title('Коэффициенты Фурье $c_k = \\langle y, \\phi_k \\rangle$')
plt.xlabel('Индекс $k$')
plt.ylabel('Значение коэффициента $c_k$ (лог. шкала)')
plt.xticks(k_values)
plt.grid(True)
plt.show()

