import numpy as np
from mpl_toolkits import mplot3d

import matplotlib.pyplot as plt

fig = plt.figure()

sample_count = 100
x_sample = 10 * np.random.random(sample_count) - 5
y_sample = 2 * x_sample - 1 + np.random.normal(0, 1.0, sample_count)

ax2 = fig.add_subplot(1, 1, 1, projection='3d')

def penalty(para_a, para_b):
    squares = (y_sample - para_a * x_sample - para_b) ** 2
    return 1 / (2 * sample_count) * np.sum(squares)

a_arr, b_arr = np.meshgrid(np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1))
func_value = np.zeros(a_arr.shape)
for x in range(a_arr.shape[0]):
    for y in range(a_arr.shape[1]):
        func_value[x, y] = penalty(a_arr[x, y], b_arr[x, y])

ax2.plot_surface(a_arr, b_arr, func_value, color='red', alpha=0.8)
ax2.set_xlabel('a parameter')
ax2.set_ylabel('b parameter')
ax2.set_zlabel('f(a, b)')

def grad(aa):
    update_vector = (y_sample - aa[0] * x_sample - aa[1])
    grad_aa = np.zeros(2)
    grad_aa[0] = -1 / sample_count * x_sample.dot(update_vector)
    grad_aa[1] = -1 / sample_count * np.sum(update_vector)
    return grad_aa

initial_aa = np.array([-3, 3])  # New initial position

learn_rates = [0.209, 0.21, 0.211]
labels = ['Low', 'Medium', 'High']
colors = ['blue', 'green', 'orange']

for lr, label, color in zip(learn_rates, labels, colors):
    aa = initial_aa.copy()
    delta = np.inf
    epsilon = 0.001
    step_count = 0
    path_a, path_b, path_f = [aa[0]], [aa[1]], [penalty(aa[0], aa[1])]
    step_count -=2
    while delta > epsilon and step_count < 100:
        aa_next = aa - lr * grad(aa)
        delta = np.linalg.norm(aa - aa_next)
        aa = aa_next
        path_a.append(aa[0])
        path_b.append(aa[1])
        path_f.append(penalty(aa[0], aa[1]))
        step_count += 1
    ax2.plot(path_a, path_b, path_f, marker='o', color=color, label=label)
    ax2.scatter(path_a[-1], path_b[-1], path_f[-1], c=color, s=80, marker='*')
    print(step_count)

ax2.legend()
plt.show()