import pandas as pd
import matplotlib.pyplot as plt


dfClasses = pd.read_csv('./imageMetadata.csv')


# Count occurrences of each label
habit_label_counts = dfClasses['Habit Label'].value_counts()
rime_label_counts = dfClasses['Rime Label'].value_counts()
# Sort rime label counts by index (alphabetically)
rime_label_counts_sorted = rime_label_counts.sort_index()
melt_label_counts = dfClasses['Melt Label'].value_counts()
entropy_values = dfClasses['Label Set Entropy'].values

# Plotting the habit label bar chart
plt.figure(figsize=(8, 6))
habit_label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Number of Habit Labels')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.savefig("HabitLabels.png")

# Plotting the habit label bar chart
plt.figure(figsize=(8, 6))
rime_label_counts_sorted.plot(kind='bar', color='red', edgecolor='black')
plt.title('Number of Rime Labels')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.savefig("RimeLabels.png")

# Plotting the habit label bar chart
plt.figure(figsize=(8, 6))
melt_label_counts.plot(kind='bar', color='green', edgecolor='black')
plt.title('Number of Melt Labels')
plt.xlabel('Labels')
plt.ylabel('Count')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=0)
plt.savefig("MeltLabels.png")

# Plotting the histogram
plt.figure(figsize=(8, 6))
plt.hist(entropy_values, bins=10, color='gray', edgecolor='black')
plt.title('Label Set Entropy')
plt.xlabel('Entropy')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig("LabelSetEntropy.png")
