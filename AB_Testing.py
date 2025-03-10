# Import necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
data = pd.read_csv('olist_order_items_dataset.csv')

# Simulate Discounts and Create A/B Groups
def create_ab_groups(df, discount=0.10):
    np.random.seed(42)
    df['group'] = np.random.choice(['Control', 'Test'], size=len(df))
    df['discounted_price'] = df['price']
    df.loc[df['group'] == 'Test', 'discounted_price'] = df['price'] * (1 - discount)
    df['profit_margin'] = df['price'] * 0.3  # Assuming a 30% profit margin
    df['discounted_margin'] = df['discounted_price'] * 0.3
    return df

data = create_ab_groups(data, discount=0.10)

# Calculate Revenue and Margin for Each Group
revenue = data.groupby('group').agg({
    'price': 'sum',
    'discounted_price': 'sum',
    'profit_margin': 'sum',
    'discounted_margin': 'sum'
}).reset_index()

print("\nRevenue Summary:\n", revenue)

# Calculate Revenue Increase Percentage
control_revenue = revenue[revenue['group'] == 'Control']['price'].values[0]
test_revenue = revenue[revenue['group'] == 'Test']['discounted_price'].values[0]

revenue_increase = ((test_revenue - control_revenue) / control_revenue) * 100
print(f"\nRevenue Increase: {revenue_increase:.2f}%")

# Check if Margins are Maintained
control_margin = revenue[revenue['group'] == 'Control']['profit_margin'].values[0]
test_margin = revenue[revenue['group'] == 'Test']['discounted_margin'].values[0]

if test_margin >= control_margin * 0.9:  # Allowing a 10% margin drop
    print("\nMargins are maintained within acceptable range.")
else:
    print("\nMargins are not maintained.")

# A/B Testing Using T-Test
control = data[data['group'] == 'Control']['price']
test = data[data['group'] == 'Test']['discounted_price']

# Perform t-test
t_stat, p_value = stats.ttest_ind(control, test)

print("\nT-Statistic:", t_stat)
print("P-Value:", p_value)

if p_value < 0.05:
    print("\nResult: Statistically significant difference detected.")
else:
    print("\nResult: No significant difference detected.")

# Visualizing the Impact of Discounts
sns.boxplot(x='group', y='discounted_price', data=data)
plt.title('Impact of Discounts on Pricing')
plt.show()

# Determine Optimal Discount Range
def optimize_discount(df, discounts):
    results = []
    for discount in discounts:
        df['discounted_price'] = df['price'] * (1 - discount)
        revenue = df['discounted_price'].sum()
        results.append({'Discount': discount, 'Revenue': revenue})
    return pd.DataFrame(results)

discount_ranges = np.arange(0, 0.5, 0.05)
optimal_discounts = optimize_discount(data, discount_ranges)

# Plot Optimal Discount Range
sns.lineplot(x='Discount', y='Revenue', data=optimal_discounts)
plt.title('Revenue vs Discount Percentage')
plt.xlabel('Discount Percentage')
plt.ylabel('Total Revenue')
plt.show()

# Save Optimal Discount Results
optimal_discounts.to_csv('data/processed/optimal_discount_ranges.csv', index=False)
print("\nOptimal discount ranges saved to 'data/processed/optimal_discount_ranges.csv'")
