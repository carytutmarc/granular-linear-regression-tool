import os
import pandas as pd
import numpy as np
from scipy.stats import linregress
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from config import FigObj, GridObj, HeatmapConfig, PlotConfig

# Load the test dataset as defined in config.py
config = PlotConfig()
df = config.load_test_dataframe()

# Load dataset from config.py
#df = config.load_dataframe()

# Create a DataFrame from PlotConfig parameters
dfprocess = pd.DataFrame({
    'Sample': df[config.sample],
    'X': df[config.xVar],
    'Y': df[config.yVar],
    'Normalization': config.normalization
})

# Global results DataFrame
results = pd.DataFrame()

def process_single_row(row, grid_vars, base_save_path):
    """Process a single row of the DataFrame to find best fit and generate plots."""
    
    x, y, sample_name, normalization = row['X'], row['Y'], row['Sample'], row['Normalization']
    
    # Find the best fit domain
    def find_best_fit_domain(x, y, start_index, increment):
        length_index = len(x)
        #print("\nLength: ", length_index, ' \n')
        results = []
    
        for start in range(start_index, length_index - increment, increment):
            for n in range(1, ((length_index - start - increment) // increment) + 1):
                length = n * increment
                end_index = start + length
    
                # Ensure that end_index does not exceed both conditions
                if end_index <= length_index and start + length < length_index:
                    X = x[start:end_index]
                    Y = y[start:end_index]
                    #print("StrtIdx: ", start, ' EndIdx: ', end_index, ' ')
                    
                    stderr = linregress(X, Y).stderr
                    inv_stderr = 1 / stderr if stderr != 0 else np.inf
                    results.append((stderr, inv_stderr, (start, end_index - 1)))
                else:
                    break
    
        return results

    # Results from best fit search
    row_results = find_best_fit_domain(x, y, grid_vars.START, grid_vars.INCREMENT)
    if not row_results:
        print(f"No results found for row index: {row.name}")
        return

    df_row_results = pd.DataFrame(row_results, columns=['stderr', 'inv_stderr', 'range'])
    df_row_results[['start', 'end']] = pd.DataFrame(df_row_results['range'].tolist(), index=df_row_results.index)
    df_row_results['length'] = df_row_results['end'] - df_row_results['start'] + 1
    pivot_table = df_row_results.pivot(index='end', columns='start', values='inv_stderr').sort_index(ascending=False)
    
    # ** New lines to identify indices of the max inv_stderr **
    max_inv_stderr = df_row_results['inv_stderr'].max()
    max_inv_stderr_indices = df_row_results.index[df_row_results['inv_stderr'] == max_inv_stderr].tolist()
    #print(f"Max inv_stderr value: {max_inv_stderr} found at indices: {max_inv_stderr_indices}")

    for idx in max_inv_stderr_indices:
        start_value = df_row_results.at[idx, 'start']
        end_value = df_row_results.at[idx, 'end']
        #print(f"Max inv_stderr value: {max_inv_stderr} found at index: {idx} with start: {start_value}, end: {end_value}")

    # Directory paths for saving figures
    heatmap_path = os.path.join(base_save_path, 'heatmaps')
    linearfit_path = os.path.join(base_save_path, 'linearfits')
    bestfit_path = os.path.join(base_save_path, 'bestfits')
    os.makedirs(heatmap_path, exist_ok=True)
    os.makedirs(linearfit_path, exist_ok=True)
    os.makedirs(bestfit_path, exist_ok=True)
    
    # Create heatmap figure and save it
    heatmap_plt = plt.figure(figsize=(8, 8))
    heatmap_config = HeatmapConfig()
    sns.heatmap(pivot_table, cmap=heatmap_config.cmap, annot=False,
                fmt='.2f', square=True, linewidths=0,
                cbar=heatmap_config.cbar, cbar_kws={'shrink': 0.75, 'aspect': 20})

    plt.title(f"{heatmap_config.title}\n{sample_name}", loc='center')
    plt.xlabel(heatmap_config.x_label)
    plt.ylabel(heatmap_config.y_label)
    
    # Initialize the filtered DataFrame with the base condition
    if grid_vars.ENDMAX is not None:
        filtered_df = df_row_results[df_row_results['end'] <= grid_vars.ENDMAX * df_row_results['end'].max()]
        filtered_df = filtered_df[(filtered_df['start'] >= grid_vars.START)]
    else:
        filtered_df = df_row_results[(df_row_results['start'] >= grid_vars.START)]

    # Apply additional filters based on defined parameters
    if grid_vars.STARTMAX is not None:
        filtered_df = filtered_df[filtered_df['start'] <= grid_vars.STARTMAX]

    if grid_vars.LENGTHMIN is not None:
        filtered_df = filtered_df[filtered_df['length'] >= grid_vars.LENGTHMIN]

    if grid_vars.LENGTHMAX is not None:
        filtered_df = filtered_df[filtered_df['length'] <= grid_vars.LENGTHMAX * filtered_df['length'].max()]

    # Calculate max_value from the resulting filtered_df
    max_value = filtered_df['inv_stderr'].max() if not filtered_df.empty else None
    
    max_index = df_row_results.index[df_row_results['inv_stderr'] == max_value].tolist()
    
    for idx in max_index:
        start_index = df_row_results.at[idx, 'start']
        end_index = df_row_results.at[idx, 'end']
    
    max_pivot = pivot_table
    max_pivot = max_pivot.where(max_pivot == max_value, np.nan)
    indices = max_pivot.stack().index.tolist() # Indices of pivot table (END, START)
    
    # Calculate conversion factor and map indices
    heatmap_x_length = pivot_table.shape[0]
    heatmap_y_length = pivot_table.shape[1]
    conversion_factor = grid_vars.INCREMENT
    heatmap_start_index = round(indices[0][1] / conversion_factor)
    heatmap_end_index = round(indices[0][0] / conversion_factor)
    
    annotation_text = (f'Max: {max_value:.2f}\n(Start: {start_index}, End: {end_index})\n '
                       f'Increment: {grid_vars.INCREMENT}')
    
    plt.gca().add_patch(plt.Rectangle((heatmap_start_index, heatmap_y_length - heatmap_end_index), 1, 1, color='red', alpha=0.8, zorder=1))
    
    plt.annotate(annotation_text,
                 xy=(0.75 * len(pivot_table.index), 0.75 * len(pivot_table.index)),
                 transform=plt.gca().transAxes,
                 color='white',
                 ha='center',
                 va='center',
                 fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.5))

    heatmap_file_path = os.path.join(heatmap_path, f"heatmap_{row.name}_{sample_name}.png")
    plt.savefig(heatmap_file_path)

    # Calculate best fit line using linear regression
    bestX = x[start_index:end_index]
    bestY = y[start_index:end_index]
    best_fit = linregress(bestX, bestY)

    # Gather the regression values
    regression_values = {
        'Sample': sample_name,
        'Index_Fit': (start_index, end_index),
        'X_Fit': bestX,
        'Y_Fit': bestY,
        'Slope': best_fit.slope,
        'Intercept': best_fit.intercept,
        'Slope_StdErr': best_fit.stderr,
        'R_value': best_fit.rvalue,
        'P_value': best_fit.pvalue,
        'Normalization': normalization,
        'Rate': best_fit.slope / normalization,
        'Rate_Uncertainty': best_fit.stderr / normalization
    }

    # Append to results using pd.concat
    global results
    results = pd.concat([results, pd.DataFrame([regression_values])], ignore_index=True)

    # Create linear fit figure and save it
    linear_plt = plt.figure(figsize=(8, 8))  # Set the size to match the combined plot
    xfit = bestX
    yfit = best_fit.slope * np.array(xfit) + best_fit.intercept

    plt.scatter(x, y, label='Data')
    plt.plot(xfit, yfit, color="red", label="Fitted Linear Part")
    plt.title(f"{config.title}\n{sample_name}", loc='center')
    plt.xlabel(config.x_label)
    plt.ylabel(config.y_label)
    plt.grid(which="major", color="gray", linestyle="-", linewidth=0.5)
    plt.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)

    # Set the y-axis to scientific notation using ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Show scientific notation for all values
    plt.gca().yaxis.set_major_formatter(formatter)

    # Add calculated values annotation
    slope, stderr = best_fit.slope, best_fit.stderr
    rate = slope / normalization
    rate_uncertainty = stderr / normalization

    plt.text(
        0.05,
        0.95,
        f"Slope = {round(best_fit.slope, 1)} ± {round(best_fit.stderr, 1)} {config.slopeunits}\n" +
        f"{config.ratelabel} = {round(rate, 1)} ± {round(rate_uncertainty, 1)} {config.normalunits}",
        backgroundcolor='white',
        transform=plt.gca().transAxes,
        verticalalignment='top'
    )

    linear_file_path = os.path.join(linearfit_path, f"linear_fit_{row.name}_{sample_name}.png")
    plt.savefig(linear_file_path)

    # Create combined figure
    combined_plt = plt.figure(figsize=(16, 8))

    # Combine heatmap subplot
    combined_ax1 = combined_plt.add_subplot(1, 2, 1)
    sns.heatmap(pivot_table, cmap=heatmap_config.cmap, annot=False,
                fmt='.2f', square=True, linewidths=0,
                cbar=heatmap_config.cbar, cbar_kws={'shrink': 0.75},
                ax=combined_ax1)

    combined_ax1.set_title(f"{heatmap_config.title}", loc='center')
    combined_ax1.set_xlabel(heatmap_config.x_label)
    combined_ax1.set_ylabel(heatmap_config.y_label)

    # Add the same heatmap annotation to the combined plot
    combined_ax1.add_patch(plt.Rectangle((heatmap_start_index, heatmap_y_length - heatmap_end_index), 1, 1, color='red', alpha=0.8, zorder=1))
    combined_ax1.annotate(annotation_text,
                 xy=(0.75 * len(pivot_table.index), 0.75 * len(pivot_table.index)),
                 transform=combined_ax1.transAxes,
                 color='white',
                 ha='center',
                 va='center',
                 fontsize=12,
                 bbox=dict(facecolor='black', alpha=0.5))

    # Combine linear fit subplot
    combined_ax2 = combined_plt.add_subplot(1, 2, 2)
    combined_ax2.scatter(x, y, label='Data')
    combined_ax2.plot(xfit, yfit, color="red", label="Fitted Linear Part")
    combined_ax2.set_title(f"{config.title}\n{sample_name}")
    combined_ax2.set_xlabel(config.x_label)
    combined_ax2.set_ylabel(config.y_label)
    combined_ax2.grid(which="major", color="gray", linestyle="-", linewidth=0.5)
    combined_ax2.grid(which="minor", color="gray", linestyle=":", linewidth=0.5)

    combined_ax2.text(
        0.05,
        0.95,
        f"Slope = {round(best_fit.slope, 1)} ± {round(best_fit.stderr, 1)} {config.slopeunits}\n" +
        f"{config.ratelabel} = {round(rate, 1)} ± {round(rate_uncertainty, 1)} {config.normalunits}",
        backgroundcolor='white',
        transform=combined_ax2.transAxes,
        verticalalignment='top'
    )

    # Set the y-axis to scientific notation using ScalarFormatter
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))  # Show scientific notation for all values
    combined_ax2.yaxis.set_major_formatter(formatter)

    # Save combined figure
    combined_file_path = os.path.join(bestfit_path, f"fig_{row.name}_best_fit_{sample_name}.png")
    plt.suptitle(f"{FigObj().title}\n{sample_name}")
    plt.tight_layout()
    plt.savefig(combined_file_path)

    # Close all figures
    plt.close(heatmap_plt)
    plt.close(linear_plt)
    plt.close(combined_plt)

# Configuration
grid_vars = GridObj()

visualization_path = os.path.join(os.getcwd(), 'visualization')
data_path = os.path.join(os.getcwd(), 'data')

# Process each row of the DataFrame
num_rows = len(dfprocess)
for index, row in dfprocess.iterrows():
    process_single_row(row, grid_vars, visualization_path)
    # Update progress
    number_complete = (index + 1)
    print(f"\rProgress: {number_complete} of {num_rows}", end='')
print()

# Append the results to the original DataFrame
final_results = pd.merge(dfprocess, results, on='Sample', how='inner')

all_results = pd.concat([df, final_results], axis=1)

# Save the results DataFrame to a CSV file if desired
final_results.to_csv(os.path.join(data_path, 'granular_linear_regression_results.csv'), index=False)

print('Finished')