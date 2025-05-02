import pandas as pd
import matplotlib.pyplot as plt

# Load the merged burglary dataset
csv_path = r"C:\Users\Gabri\Documents\GitHub\CBL---group-23\all_burglary.csv"
df = pd.read_csv(csv_path, dtype=str)

# Parse Month column as datetime
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')

# Aggregate counts of burglaries per month
monthly_counts = df.groupby('Month').size().reset_index(name='Burglaries')

# Extract month number for grouping
monthly_counts['month_number'] = monthly_counts['Month'].dt.month

# Prepare data for seasonal boxplot
monthly_groups = [
    monthly_counts.loc[monthly_counts['month_number'] == m, 'Burglaries'].values
    for m in range(1, 13)
]

# Month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create boxplot with styling matching your example
fig, ax = plt.subplots(figsize=(12, 6))
boxprops = dict(linewidth=1.5)
medianprops = dict(linewidth=2.0, color='black')
colors = plt.cm.tab20.colors[:12]

bp = ax.boxplot(monthly_groups, patch_artist=True,
                boxprops=boxprops, medianprops=medianprops)

# Apply colors to boxes and set transparency
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Styling
ax.set_title('Seasonal Distribution of Monthly Residential Burglaries in London', fontsize=16, pad=15)
ax.set_xlabel('Month', fontsize=12)
ax.set_ylabel('Number of Burglaries', fontsize=12)
ax.set_xticklabels(month_labels)
ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

plt.tight_layout()
plt.show()
