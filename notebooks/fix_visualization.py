import json

# Read the notebook
with open('03_model_training.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the problematic cell (it's the one with the error)
# The visualization cell is after the model saving cell
# Looking for the cell with matplotlib code

fixed_code = """import matplotlib.pyplot as plt

# Model names for x-axis
model_names = ['Random Forest', 'XGBoost', 'Gradient Boosting']

# Metric values for each model
rmse_scores = [rf_rmse, xgb_rmse, gb_rmse]
mae_scores = [rf_mae, xgb_mae, gb_mae]
r2_scores = [rf_r2, xgb_r2, gb_r2]

# Improved plot design
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)

metrics = [
    ('RMSE', rmse_scores, 'deepskyblue'),
    ('MAE', mae_scores, 'mediumseagreen'),
    ('R²', r2_scores, 'tomato')
]

for ax, (title, scores, color) in zip(axes, metrics):
    bars = ax.bar(model_names, scores, color=color, edgecolor='black')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_xticklabels(model_names, rotation=15, fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    # Annotate bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', fontsize=11)

fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()"""

# Find and fix the cell with the visualization code
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and 'source' in cell:
        source = ''.join(cell['source'])
        if 'Model Performance Comparison' in source and 'model_names' in source:
            # This is the cell with the error
            notebook['cells'][i]['source'] = fixed_code.split('\n')
            notebook['cells'][i]['outputs'] = []
            notebook['cells'][i]['execution_count'] = None
            print(f"Fixed cell {i}")
            break

# Write the fixed notebook
with open('03_model_training.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2)

print("Notebook fixed successfully!")
