import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file
df = pd.read_csv('fault_injection_log.csv')

# Filter out rows where Initial_Value equals Final_Classification
df_filtered = df[df['Initial_Value'] != df['Final_Classification']]

# Group by Perturbed_Data_Type and count occurrences
perturbed_counts = df_filtered['Perturbed_Data_Type'].value_counts()

# Plot the results
plt.figure(figsize=(10, 6))
plt.bar(perturbed_counts.index, perturbed_counts.values)
plt.xlabel('Perturbed Data Type')
plt.ylabel('Number of Bit Flips')
plt.title('Dependency of Perturbed Data Type on Bit Flips in Lenet5')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
