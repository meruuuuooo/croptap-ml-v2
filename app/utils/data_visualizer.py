
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Not directly used here, but good to import

def visualize_province_data_distribution(data_dir="raw_datasets", output_dir="plots"):
    """
    Generates bar charts showing the distribution of data points per province
    from soil_test_data.csv and historical_crop_performance.csv.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Process soil_test_data.csv ---
    soil_data_path = os.path.join(data_dir, "soil_test_data.csv")
    try:
        soil_df = pd.read_csv(soil_data_path)
        soil_province_counts = soil_df['province'].value_counts().reset_index()
        soil_province_counts.columns = ['Province', 'Count']

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='Province', data=soil_province_counts.head(20), palette='viridis')
        plt.title('Top 20 Provinces by Soil Test Data Points')
        plt.xlabel('Number of Soil Test Data Points')
        plt.ylabel('Province')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'soil_data_province_distribution.png'))
        plt.close()
        print(f"Generated plot: {os.path.join(output_dir, 'soil_data_province_distribution.png')}")
    except FileNotFoundError:
        print(f"Error: {soil_data_path} not found.")
    except Exception as e:
        print(f"Error processing {soil_data_path}: {e}")


    # --- Process historical_crop_performance.csv ---
    performance_data_path = os.path.join(data_dir, "historical_crop_performance.csv")
    try:
        performance_df = pd.read_csv(performance_data_path)
        # Ensure consistent column name 'Province' for plotting
        if 'Province' in performance_df.columns:
            performance_province_counts = performance_df['Province'].value_counts().reset_index()
            performance_province_counts.columns = ['Province', 'Count']
        elif 'province' in performance_df.columns:
            performance_province_counts = performance_df['province'].value_counts().reset_index()
            performance_province_counts.columns = ['Province', 'Count']
        else:
            print(f"Error: 'Province' or 'province' column not found in {performance_data_path}.")
            return

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Count', y='Province', data=performance_province_counts.head(20), palette='plasma')
        plt.title('Top 20 Provinces by Historical Crop Performance Data Points')
        plt.xlabel('Number of Historical Crop Performance Data Points')
        plt.ylabel('Province')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'crop_performance_province_distribution.png'))
        plt.close()
        print(f"Generated plot: {os.path.join(output_dir, 'crop_performance_province_distribution.png')}")

    except FileNotFoundError:
        print(f"Error: {performance_data_path} not found.")
    except Exception as e:
        print(f"Error processing {performance_data_path}: {e}")

def visualize_soil_npk_distribution(data_dir="raw_datasets", output_dir="plots"):
    """
    Generates bar charts for the distribution of Nitrogen, Phosphorus, and Potassium
    from soil_test_data.csv.
    Assumes NPK levels are categorical (Low, Medium, High).
    """
    os.makedirs(output_dir, exist_ok=True)
    soil_data_path = os.path.join(data_dir, "soil_test_data.csv")

    try:
        soil_df = pd.read_csv(soil_data_path)
        npk_columns = ['nitrogen', 'phosphorus', 'potassium']
        
        for col in npk_columns:
            if col in soil_df.columns:
                # Order categories for consistent plotting
                category_order = ['Low', 'Medium', 'High']
                # Filter out any categories not in the order, then reindex
                counts = soil_df[col].value_counts().reindex(category_order).fillna(0)

                plt.figure(figsize=(8, 6))
                sns.barplot(x=counts.index, y=counts.values, palette='coolwarm', order=category_order)
                plt.title(f'Distribution of Soil {col.capitalize()} Levels')
                plt.xlabel(f'{col.capitalize()} Level')
                plt.ylabel('Number of Samples')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'soil_{col}_distribution.png'))
                plt.close()
                print(f"Generated plot: {os.path.join(output_dir, f'soil_{col}_distribution.png')}")
            else:
                print(f"Column '{col}' not found in {soil_data_path}.")
    except FileNotFoundError:
        print(f"Error: {soil_data_path} not found.")
    except Exception as e:
        print(f"Error processing {soil_data_path}: {e}")

def visualize_climate_data_distribution(data_dir="raw_datasets", output_dir="plots"):
    """
    Generates histograms for the annual distribution of various climate parameters
    from climate_data.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    climate_data_path = os.path.join(data_dir, "climate_data.csv")

    try:
        climate_df = pd.read_csv(climate_data_path)
        
        # Get unique climate parameters
        parameters = climate_df['PARAMETER'].unique()

        for param in parameters:
            param_df = climate_df[climate_df['PARAMETER'] == param]
            
            # Plot distribution of ANNUAL values
            plt.figure(figsize=(10, 6))
            sns.histplot(param_df['ANNUAL'], kde=True, bins=30)
            plt.title(f'Distribution of Annual {param.replace("_", " ").title()}')
            plt.xlabel(f'Annual {param.replace("_", " ").title()} Value')
            plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'climate_annual_{param}.png'))
            plt.close()
            print(f"Generated plot: {os.path.join(output_dir, f'climate_annual_{param}.png')}")

            # Plot monthly distributions as box plots
            monthly_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
            if all(col in param_df.columns for col in monthly_cols):
                # Melt the DataFrame to long format for easier plotting
                monthly_data = param_df[monthly_cols].melt(var_name='Month', value_name='Value')
                
                # Order months correctly
                month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
                monthly_data['Month'] = pd.Categorical(monthly_data['Month'], categories=month_order, ordered=True)
                monthly_data = monthly_data.sort_values('Month')

                plt.figure(figsize=(14, 7))
                sns.boxplot(x='Month', y='Value', data=monthly_data, palette='coolwarm')
                plt.title(f'Monthly Distribution of {param.replace("_", " ").title()}')
                plt.xlabel('Month')
                plt.ylabel('Value')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'climate_monthly_{param}.png'))
                plt.close()
                print(f"Generated plot: {os.path.join(output_dir, f'climate_monthly_{param}.png')}")


    except FileNotFoundError:
        print(f"Error: {climate_data_path} not found.")
    except Exception as e:
        print(f"Error processing {climate_data_path}: {e}")

def visualize_historical_crop_performance_distribution(data_dir="raw_datasets", output_dir="plots"):
    """
    Generates visualizations for historical crop performance data.
    - Histograms for overall Volume_Production and Area_Planted_Harvested.
    - Bar charts for top N crops by total Volume_Production and Area_Planted_Harvested.
    """
    os.makedirs(output_dir, exist_ok=True)
    performance_data_path = os.path.join(data_dir, "historical_crop_performance.csv")

    try:
        performance_df = pd.read_csv(performance_data_path)

        # Plot histograms for numerical distributions
        for col in ['Volume_Production', 'Area_Planted_Harvested']:
            if col in performance_df.columns:
                plt.figure(figsize=(10, 6))
                sns.histplot(performance_df[col], kde=True, bins=50)
                plt.title(f'Distribution of {col.replace("_", " ").title()}')
                plt.xlabel(col.replace("_", " ").title())
                plt.ylabel('Frequency')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'historical_crop_{col}_distribution.png'))
                plt.close()
                print(f"Generated plot: {os.path.join(output_dir, f'historical_crop_{col}_distribution.png')}")
            else:
                print(f"Column '{col}' not found in {performance_data_path}.")

        # Plot top N crops by aggregated metrics
        n_top_crops = 15 # Can be adjusted
        
        # Aggregate by Crop for Volume_Production
        crop_production = performance_df.groupby('Crop')['Volume_Production'].sum().nlargest(n_top_crops).reset_index()
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Volume_Production', y='Crop', data=crop_production, palette='cubehelix')
        plt.title(f'Top {n_top_crops} Crops by Total Volume Production')
        plt.xlabel('Total Volume Production')
        plt.ylabel('Crop')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'historical_crop_top_{n_top_crops}_production.png'))
        plt.close()
        print(f"Generated plot: {os.path.join(output_dir, f'historical_crop_top_{n_top_crops}_production.png')}")

        # Aggregate by Crop for Area_Planted_Harvested
        crop_area = performance_df.groupby('Crop')['Area_Planted_Harvested'].sum().nlargest(n_top_crops).reset_index()
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Area_Planted_Harvested', y='Crop', data=crop_area, palette='cubehelix')
        plt.title(f'Top {n_top_crops} Crops by Total Area Planted/Harvested')
        plt.xlabel('Total Area Planted/Harvested')
        plt.ylabel('Crop')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'historical_crop_top_{n_top_crops}_area.png'))
        plt.close()
        print(f"Generated plot: {os.path.join(output_dir, f'historical_crop_top_{n_top_crops}_area.png')}")

    except FileNotFoundError:
        print(f"Error: {performance_data_path} not found.")
    except Exception as e:
        print(f"Error processing {performance_data_path}: {e}")

def visualize_recommendation_score_breakdown(crop_name: str, rule_based_score: float, ml_score: float, hybrid_score: float, output_dir="plots"):
    """
    Generates a bar chart showing the breakdown of the hybrid score for a given crop recommendation.
    """
    os.makedirs(output_dir, exist_ok=True)

    data = {
        'Component': ['Rule-Based Score (40%)', 'ML Model Score (60%)', 'Hybrid Score'],
        'Score': [rule_based_score, ml_score, hybrid_score]
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Component', y='Score', data=df, palette='viridis')
    plt.title(f'Recommendation Score Breakdown for {crop_name}')
    plt.ylim(0, 100) # Assuming scores are between 0 and 100
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'recommendation_score_breakdown_{crop_name.replace(" ", "_")}.png'))
    plt.close()
    print(f"Generated plot: {os.path.join(output_dir, f'recommendation_score_breakdown_{crop_name.replace(" ", "_")}.png')}")


def visualize_correlation_matrix(data_dir="raw_datasets", output_dir="plots"):
    """
    Generates correlation matrix heatmaps for numerical columns in climate_data.csv.
    """
    os.makedirs(output_dir, exist_ok=True)
    climate_data_path = os.path.join(data_dir, "climate_data.csv")

    try:
        climate_df = pd.read_csv(climate_data_path)
        parameters = climate_df['PARAMETER'].unique()
        
        monthly_cols = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

        for param in parameters:
            param_df = climate_df[climate_df['PARAMETER'] == param]
            
            # Select only numerical columns for correlation, including ANNUAL
            numerical_cols = monthly_cols + ['ANNUAL']
            param_numerical_df = param_df[numerical_cols].dropna()

            if not param_numerical_df.empty and len(param_numerical_df.columns) > 1:
                correlation_matrix = param_numerical_df.corr()
                
                plt.figure(figsize=(12, 10))
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
                plt.title(f'Correlation Matrix for {param.replace("_", " ").title()} (Monthly and Annual)')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'correlation_matrix_{param}.png'))
                plt.close()
                print(f"Generated plot: {os.path.join(output_dir, f'correlation_matrix_{param}.png')}")
            else:
                print(f"Not enough numerical data for correlation matrix for parameter: {param}")

    except FileNotFoundError:
        print(f"Error: {climate_data_path} not found.")
    except Exception as e:
        print(f"Error processing {climate_data_path} for correlation matrix: {e}")

def visualize_crop_requirements(data_dir="raw_datasets", output_dir="plots"):
    """
    Generates visualizations for crop NPK and climate requirements.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- NPK Requirements ---
    npk_path = os.path.join(data_dir, "crop_npk_level_requirements.csv")
    try:
        npk_df = pd.read_csv(npk_path)
        
        npk_levels = ['n_requirement', 'p_requirement', 'k_requirement']
        category_order = ['low', 'medium', 'high']

        for npk_col in npk_levels:
            if npk_col in npk_df.columns:
                counts = npk_df[npk_col].value_counts().reindex(category_order).fillna(0)
                plt.figure(figsize=(8, 6))
                sns.barplot(x=counts.index, y=counts.values, palette='Greens', order=category_order)
                plt.title(f'Distribution of Crop {npk_col.replace("_", " ").title()}')
                plt.xlabel(f'{npk_col.replace("_", " ").title()} Level')
                plt.ylabel('Number of Crops')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'crop_{npk_col}_distribution.png'))
                plt.close()
                print(f"Generated plot: {os.path.join(output_dir, f'crop_{npk_col}_distribution.png')}")
            else:
                print(f"Column '{npk_col}' not found in {npk_path}.")

    except FileNotFoundError:
        print(f"Error: {npk_path} not found.")
    except Exception as e:
        print(f"Error processing {npk_path}: {e}")

    # --- Climate Requirements ---
    climate_req_path = os.path.join(data_dir, "crop_climate_requirements.csv")
    try:
        climate_req_df = pd.read_csv(climate_req_path)
        
        # For simplicity, let's visualize the average of min/max for a selection of crops
        climate_req_df['Avg_Temp_C'] = (climate_req_df['Temp_Min_C'] + climate_req_df['Temp_Max_C']) / 2
        climate_req_df['Avg_Rainfall_mm'] = (climate_req_df['Rainfall_Min_mm'] + climate_req_df['Rainfall_Max_mm']) / 2
        climate_req_df['Avg_Humidity_Percent'] = (climate_req_df['Humidity_Min_Percent'] + climate_req_df['Humidity_Max_Percent']) / 2

        # Select top N crops by some metric, or just a sample
        # For now, let's take a sample of top 10 crops alphabetically to show comparison
        # (Alternatively, could base this on top producing crops from historical data, but that's a join)
        sample_crops = climate_req_df['Crop_Name'].sample(n=min(10, len(climate_req_df)), random_state=42)
        sample_df = climate_req_df[climate_req_df['Crop_Name'].isin(sample_crops)].set_index('Crop_Name')

        # Plotting climate requirements as grouped bar charts
        climate_params = ['Avg_Temp_C', 'Avg_Rainfall_mm', 'Avg_Humidity_Percent']
        for param in climate_params:
            plt.figure(figsize=(12, 7))
            sns.barplot(x=sample_df.index, y=param, data=sample_df, palette='YlGnBu')
            plt.title(f'{param.replace("_", " ").title()} Requirements for Sample Crops')
            plt.xlabel('Crop Name')
            plt.ylabel(param.replace("_", " ").title())
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'crop_avg_{param}_requirements.png'))
            plt.close()
            print(f"Generated plot: {os.path.join(output_dir, f'crop_avg_{param}_requirements.png')}")


    except FileNotFoundError:
        print(f"Error: {climate_req_path} not found.")
    except Exception as e:
        print(f"Error processing {climate_req_path}: {e}")

def visualize_model_performance_comparison(model_metrics: dict, output_dir="plots"):
    """
    Generates a grouped bar chart comparing the performance metrics (RMSE, MAE, R2)
    of different models.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert dictionary to DataFrame for easy plotting
    metrics_list = []
    for model_name, metrics in model_metrics.items():
        metrics_list.append({
            'Model': model_name,
            'Metric': 'RMSE',
            'Value': metrics['RMSE']
        })
        metrics_list.append({
            'Model': model_name,
            'Metric': 'MAE',
            'Value': metrics['MAE']
        })
        metrics_list.append({
            'Model': model_name,
            'Metric': 'R²',
            'Value': metrics['R²']
        })
    df_metrics = pd.DataFrame(metrics_list)

    plt.figure(figsize=(12, 7))
    sns.barplot(x='Metric', y='Value', hue='Model', data=df_metrics, palette='tab10')
    plt.title('Model Performance Comparison (RMSE, MAE, R²)')
    plt.ylabel('Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_performance_comparison.png'))
    plt.close()
    print(f"Generated plot: {os.path.join(output_dir, 'model_performance_comparison.png')}")

def visualize_predicted_actual_heatmap(models_dir="models", output_dir="plots"):
    """
    Generates a 2D histogram (heatmap) of predicted vs actual suitability scores
    for the best model.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load datasets and model info
        val_df = pd.read_csv(os.path.join(models_dir, "validation_dataset.csv"))
        with open(os.path.join(models_dir, "model_info.json"), "r") as f:
            model_info = json.load(f)

        feature_columns = model_info['feature_names']
        X_val = val_df[feature_columns]
        y_val = val_df['suitability_score']

        # Load the best model
        best_model_path = os.path.join(models_dir, "crop_suitability_model.pkl")
        best_model = joblib.load(best_model_path)
        best_model_name = model_info.get('model_type', 'Unknown')

        y_pred = best_model.predict(X_val)

        plt.figure(figsize=(10, 8))
        # Use histplot with 2D functionality for a heatmap
        sns.histplot(x=y_val, y=y_pred, bins=50, pthresh=.1, cmap="viridis", cbar=True)
        plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2) # Ideal line
        plt.xlabel('Actual Suitability Score')
        plt.ylabel('Predicted Suitability Score')
        plt.title(f'Predicted vs Actual Suitability Score Heatmap for {best_model_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'predicted_actual_heatmap_{best_model_name.replace(" ", "_")}.png'))
        plt.close()
        print(f"Generated plot: {os.path.join(output_dir, f'predicted_actual_heatmap_{best_model_name.replace(" ", "_")}.png')}")

    except FileNotFoundError as e:
        print(f"Error loading model/data for predicted vs actual heatmap: {e}")
    except Exception as e:
        print(f"Error generating predicted vs actual heatmap: {e}")


if __name__ == "__main__":
    visualize_province_data_distribution()
    visualize_soil_npk_distribution()
    visualize_climate_data_distribution()
    visualize_historical_crop_performance_distribution()

    # Hypothetical recommendation for visualization
    # The hybrid scoring is weighted, with 40% of the score coming from the rule-based system and 60% from the machine learning model.
    hypothetical_crop = "Rice"
    hypothetical_rule_score = 70
    hypothetical_ml_score = 85
    hypothetical_hybrid_score = (0.40 * hypothetical_rule_score) + (0.60 * hypothetical_ml_score)
    visualize_recommendation_score_breakdown(
        crop_name=hypothetical_crop,
        rule_based_score=hypothetical_rule_score,
        ml_score=hypothetical_ml_score,
        hybrid_score=hypothetical_hybrid_score
    )
    visualize_correlation_matrix()
    visualize_crop_requirements()

    # Model performance metrics extracted from notebooks/03_model_training.ipynb
    model_metrics_data = {
        'RandomForest': {'RMSE': 11.55, 'MAE': 6.14, 'R²': 0.7481},
        'XGBoost': {'RMSE': 11.57, 'MAE': 6.23, 'R²': 0.7470},
        'GradientBoosting': {'RMSE': 11.56, 'MAE': 6.25, 'R²': 0.7476}
    }
    visualize_model_performance_comparison(model_metrics_data)
    visualize_predicted_actual_heatmap()
