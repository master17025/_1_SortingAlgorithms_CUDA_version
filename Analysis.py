
import pandas as pd
import matplotlib.pyplot as plt

# Load data from CSV files
counting_sort_data = pd.read_csv('counting_sort.csv')
merge_sort_data = pd.read_csv('merge_sort.csv')


#counting_sort_data = pd.read_csv('counting_sort_CopyDevice.csv')
#merge_sort_data = pd.read_csv('merge_sort_CopyDevice.csv')

# Plot Counting Sort data
plt.figure(figsize=(10, 5))
plt.plot(counting_sort_data['Number of Elements'], counting_sort_data['CPU Time (ms)'], label='CPU Time (ms)', marker='o')
plt.plot(counting_sort_data['Number of Elements'], counting_sort_data['GPU Time (ms)'], label='GPU Time (ms)', marker='o')
plt.xlabel('Number of Elements')
plt.ylabel('Time (ms)')
plt.title('Counting Sort: CPU vs GPU Time')
plt.legend()
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True)
plt.savefig('counting_sort_plot.png')
plt.show()

# Plot Merge Sort data
plt.figure(figsize=(10, 5))
plt.plot(merge_sort_data['Number of Elements'], merge_sort_data['CPU Time (ms)'], label='CPU Time (ms)', marker='o')
plt.plot(merge_sort_data['Number of Elements'], merge_sort_data['GPU Time (ms)'], label='GPU Time (ms)', marker='o')
plt.xlabel('Number of Elements')
plt.ylabel('Time (ms)')
plt.title('Merge Sort: CPU vs GPU Time')
plt.legend()
plt.xscale('log', base=2)
plt.yscale('log')
plt.grid(True)
plt.savefig('merge_sort_plot.png')
plt.show()
