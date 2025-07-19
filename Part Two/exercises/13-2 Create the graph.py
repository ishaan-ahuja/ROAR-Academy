import matplotlib.pyplot as plt

# Sample data
x = [1.0, 2.0, 3.0]
y = [2.0, 4.0, 1.0]

plt.plot(x, y, marker='o')

plt.title('Sample Linear Graph')
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.xlim(1.0, 3.0)
plt.ylim(1.0, 4.0)


plt.grid(False)
plt.show()