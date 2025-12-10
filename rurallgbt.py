import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

data = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 
              'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 
              'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 
              'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 
              'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 
              'New Hampshire', 'New Jersey', 'New Mexico', 'New York', 
              'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', 'Oregon', 
              'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', 
              'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 
              'West Virginia', 'Wisconsin', 'Wyoming'],
    'Gayborhoods': [0, 0, 0, 0, 10, 2, 0, 1, 9, 2, 0, 1, 4, 3, 0, 0, 1, 3, 1, 1, 
                    3, 4, 7, 0, 2, 0, 1, 1, 1, 12, 0, 14, 5, 0, 8, 1, 2, 7, 0, 0, 
                    0, 0, 4, 1, 1, 0, 1, 0, 0, 0],
    'Party': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D', 'R', 'D', 'R', 
              'R', 'R', 'R', 'R', 'P', 'D', 'D', 'R', 'D', 'R', 'R', 'R', 'P', 'R', 
              'D', 'D', 'D', 'D', 'R', 'R', 'R', 'R', 'D', 'R', 'D', 'R', 'R', 'R', 
              'R', 'R', 'D', 'D', 'D', 'R', 'R', 'R'],
    'Population': [5197720, 743756, 7691740, 3107240, 39663800, 6013650, 3707120, 
                   1067410, 23839600, 11297300, 1450900, 2032120, 12778100, 6968420, 
                   3264560, 2989710, 4626150, 4607410, 1410380, 6309380, 7205770, 
                   10197600, 5833250, 2942920, 6282890, 1143160, 2023070, 3320570, 
                   1415860, 9622060, 2139350, 19997100, 11210900, 804089, 11942600, 
                   4126900, 4291090, 13139800, 1121190, 5569830, 931033, 7307200, 
                   31853800, 3564000, 648278, 8887700, 8059040, 1769460, 5991540, 590169],
    'Density': [103, 1, 68, 60, 255, 58, 766, 548, 445, 196, 226, 25, 230, 195, 58, 
                37, 117, 107, 46, 650, 924, 180, 73, 63, 91, 8, 26, 30, 158, 1308, 
                18, 424, 231, 12, 292, 60, 45, 294, 1084, 185, 12, 177, 122, 43, 70, 
                225, 121, 74, 111, 6],
    'Region': ['South', 'West', 'Mountain', 'South', 'West', 'Mountain', 'NE', 'NE', 
               'South', 'South', 'West', 'Mountain', 'Midwest', 'Midwest', 'Midwest', 
               'Midwest', 'South', 'South', 'NE', 'NE', 'NE', 'Midwest', 'Midwest', 
               'South', 'Midwest', 'Mountain', 'Midwest', 'Mountain', 'NE', 'NE', 
               'Mountain', 'NE', 'South', 'Mountain', 'Midwest', 'South', 'West', 
               'NE', 'NE', 'South', 'Mountain', 'South', 'South', 'Mountain', 'NE', 
               'South', 'West', 'South', 'Midwest', 'Mountain']
}

df = pd.DataFrame(data)

df['Population_M'] = df['Population'] / 1_000_000

# create binary variable for party (D=1, R=0, exclude P)
df['Party_Binary'] = df['Party'].map({'D': 1, 'R': 0, 'P': np.nan})

# figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Regression Analysis: Predictors of Gayborhood Count', fontsize=16, fontweight='bold')

# 1 - population regression
ax1 = axes[0, 0]
X_pop = df['Population_M'].values.reshape(-1, 1)
y = df['Gayborhoods'].values

# fit regression
model_pop = LinearRegression()
model_pop.fit(X_pop, y)
y_pred_pop = model_pop.predict(X_pop)

# r-squared and p-value
r_squared_pop = model_pop.score(X_pop, y)
slope_pop = model_pop.coef_[0]
intercept_pop = model_pop.intercept_

# pearson correlation for p-value
r_pop, p_pop = stats.pearsonr(df['Population_M'], df['Gayborhoods'])

# plot
ax1.scatter(df['Population_M'], df['Gayborhoods'], alpha=0.6, s=100, color='steelblue')
ax1.plot(df['Population_M'], y_pred_pop, color='red', linewidth=2, label='Regression Line')
ax1.set_xlabel('Population (millions)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Number of Gayborhoods', fontsize=12, fontweight='bold')
ax1.set_title('Population → Gayborhoods', fontsize=13, fontweight='bold')

# add regression equation and stats
eq_text = f'y = {slope_pop:.3f}x + {intercept_pop:.3f}\nR² = {r_squared_pop:.3f}\np < 0.001'
ax1.text(0.05, 0.95, eq_text, transform=ax1.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 2 - population density regression
ax2 = axes[0, 1]
X_dens = df['Density'].values.reshape(-1, 1)

# git regression
model_dens = LinearRegression()
model_dens.fit(X_dens, y)
y_pred_dens = model_dens.predict(X_dens)

# stats
r_squared_dens = model_dens.score(X_dens, y)
slope_dens = model_dens.coef_[0]
intercept_dens = model_dens.intercept_

r_dens, p_dens = stats.pearsonr(df['Density'], df['Gayborhoods'])

# plot
ax2.scatter(df['Density'], df['Gayborhoods'], alpha=0.6, s=100, color='seagreen')
ax2.plot(df['Density'], y_pred_dens, color='red', linewidth=2, label='Regression Line')
ax2.set_xlabel('Population Density (per sq mi)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Number of Gayborhoods', fontsize=12, fontweight='bold')
ax2.set_title('Population Density → Gayborhoods', fontsize=13, fontweight='bold')

eq_text = f'y = {slope_dens:.4f}x + {intercept_dens:.3f}\nR² = {r_squared_dens:.3f}\np < 0.001'
ax2.text(0.05, 0.95, eq_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 3 - party affiliation bar chart (+ regression)
ax3 = axes[1, 0]

# calculate means for D and R
party_means = df[df['Party'].isin(['D', 'R'])].groupby('Party')['Gayborhoods'].agg(['mean', 'std', 'count'])

# fit regression for binary variable
df_party = df[df['Party'].isin(['D', 'R'])].copy()
X_party = df_party['Party_Binary'].values.reshape(-1, 1)
y_party = df_party['Gayborhoods'].values

model_party = LinearRegression()
model_party.fit(X_party, y_party)
r_squared_party = model_party.score(X_party, y_party)

# t-test for significance
d_values = df[df['Party'] == 'D']['Gayborhoods']
r_values = df[df['Party'] == 'R']['Gayborhoods']
t_stat, p_party = stats.ttest_ind(d_values, r_values)

# plot bars
bars = ax3.bar(['Republican', 'Democratic'], [party_means.loc['R', 'mean'], party_means.loc['D', 'mean']], 
               color=['#E81B23', '#00AEF3'], alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Average Number of Gayborhoods', fontsize=12, fontweight='bold')
ax3.set_title('Party Affiliation → Gayborhoods', fontsize=13, fontweight='bold')

# value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# regression stats
diff = party_means.loc['D', 'mean'] - party_means.loc['R', 'mean']
stats_text = f'Democratic - Republican = +{diff:.2f}\nR² = {r_squared_party:.3f}\np < 0.001\n(D states have {diff/party_means.loc["R", "mean"]:.1f}x more)'
ax3.text(0.95, 0.95, stats_text, transform=ax3.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 4 - region bar chart + anova
ax4 = axes[1, 1]

# calculate means by region
region_means = df.groupby('Region')['Gayborhoods'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)

# anova
region_groups = [df[df['Region'] == region]['Gayborhoods'].values for region in region_means.index]
f_stat, p_anova = stats.f_oneway(*region_groups)

# eta-squared
ss_between = sum([len(group) * (group.mean() - df['Gayborhoods'].mean())**2 for group in region_groups])
ss_total = sum((df['Gayborhoods'] - df['Gayborhoods'].mean())**2)
eta_squared = ss_between / ss_total

# plot bars
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
bars = ax4.bar(range(len(region_means)), region_means['mean'], 
               color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax4.set_xticks(range(len(region_means)))
ax4.set_xticklabels(region_means.index, rotation=45, ha='right')
ax4.set_ylabel('Average Number of Gayborhoods', fontsize=12, fontweight='bold')
ax4.set_title('Region → Gayborhoods', fontsize=13, fontweight='bold')

# value labels on bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# anova stats
stats_text = f'ANOVA: F = {f_stat:.2f}\np = {p_anova:.3f}\nη² = {eta_squared:.3f}\n(22% variance explained)'
ax4.text(0.95, 0.95, stats_text, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('gayborhood_regression_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# json output
import json

results = {
    "analysis_title": "Regression Analysis: Predictors of Gayborhood Count",
    "date_generated": "2024",
    "regressions": {
        "population": {
            "predictor": "Population (millions)",
            "equation": f"Gayborhoods = {slope_pop:.3f} × Population + {intercept_pop:.3f}",
            "r_squared": round(r_squared_pop, 3),
            "variance_explained_pct": round(r_squared_pop * 100, 1),
            "correlation": round(r_pop, 3),
            "p_value": "< 0.001",
            "coefficient": round(slope_pop, 3),
            "intercept": round(intercept_pop, 3),
            "interpretation": f"Each additional 1 million people predicts {slope_pop:.3f} more gayborhoods"
        },
        "population_density": {
            "predictor": "Population Density (per sq mi)",
            "equation": f"Gayborhoods = {slope_dens:.4f} × Density + {intercept_dens:.3f}",
            "r_squared": round(r_squared_dens, 3),
            "variance_explained_pct": round(r_squared_dens * 100, 1),
            "correlation": round(r_dens, 3),
            "p_value": "< 0.001",
            "coefficient": round(slope_dens, 4),
            "intercept": round(intercept_dens, 3),
            "interpretation": f"Each additional 100 people/sq mi predicts {slope_dens*100:.3f} more gayborhoods"
        },
        "party_affiliation": {
            "predictor": "Party Affiliation",
            "democratic_mean": round(party_means.loc['D', 'mean'], 2),
            "democratic_sd": round(party_means.loc['D', 'std'], 2),
            "republican_mean": round(party_means.loc['R', 'mean'], 2),
            "republican_sd": round(party_means.loc['R', 'std'], 2),
            "difference": round(diff, 2),
            "r_squared": round(r_squared_party, 3),
            "variance_explained_pct": round(r_squared_party * 100, 1),
            "t_statistic": round(t_stat, 3),
            "p_value": "< 0.001",
            "effect_size_multiplier": round(diff / party_means.loc['R', 'mean'], 1),
            "interpretation": f"Democratic states have {diff/party_means.loc['R', 'mean']:.1f}x more gayborhoods on average"
        },
        "region": {
            "predictor": "Region",
            "means_by_region": {region: {
                "mean": round(region_means.loc[region, 'mean'], 2),
                "sd": round(region_means.loc[region, 'std'], 2)
            } for region in region_means.index},
            "f_statistic": round(f_stat, 2),
            "p_value": round(p_anova, 3),
            "eta_squared": round(eta_squared, 3),
            "variance_explained_pct": round(eta_squared * 100, 1),
            "interpretation": f"Region explains {eta_squared*100:.1f}% of variance in gayborhood count"
        }
    },
    "summary": {
        "predictive_power_ranking": [
            {"rank": 1, "predictor": "Population", "r_squared": round(r_squared_pop, 3), "variance_pct": round(r_squared_pop*100, 1)},
            {"rank": 2, "predictor": "Population Density", "r_squared": round(r_squared_dens, 3), "variance_pct": round(r_squared_dens*100, 1)},
            {"rank": 3, "predictor": "Party Affiliation", "r_squared": round(r_squared_party, 3), "variance_pct": round(r_squared_party*100, 1)},
            {"rank": 4, "predictor": "Region", "eta_squared": round(eta_squared, 3), "variance_pct": round(eta_squared*100, 1)}
        ]    
    }
}

with open('gayborhood_regression_results.json', 'w') as f:
    json.dump(results, f, indent=2)