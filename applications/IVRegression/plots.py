import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator, NullFormatter, FixedLocator, ScalarFormatter

import numpy as np
import os
import json

def plot_line_loss_from_json(json_paths, column_name, metrics_file="metrics", vertical_axis_scale="log", y_axis_name="Loss", labels=None, colors=None):
    """
    Plot the average loss curve with standard deviation from JSON files over iterations.
    :param json_paths: list of paths to directories containing JSON files
    :param column_name: name of the column to plot
    """
    plt.clf()
    plt.figure(figsize=(6, 4))

    # Method's data directory
    for i, dir_path in enumerate(json_paths):
        all_runs_data = []
        # One run's data directory
        for run_directory in os.listdir(dir_path):
            run_path = os.path.join(dir_path, run_directory)
            # Path to the metrics file
            if os.path.isdir(run_path):
                metrics_path = os.path.join(run_path, "metrics", metrics_file+".json")

                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as json_file:
                        data = [json.loads(line) for line in json_file]
                        if data == [] or all([elem is None for elem in data]):
                            continue
                        all_runs_data.append(data)

        # Skip if no data was found
        if all_runs_data == [] or all([elem is None for elem in all_runs_data]):
            continue

        # Calculate the average and standard deviation over all runs
        iterations = [entry['iter'] for entry in all_runs_data[0]]
        all_runs_values = np.array([[entry.get(column_name, None) for entry in run_data] for run_data in all_runs_data])

        # Filter out None values
        all_runs_values = [run_values for run_values in all_runs_values if None not in run_values]

        # If all_runs_values is empty after filtering, skip this iteration
        if not all_runs_values:
            continue
        else:
            mean_values = np.mean(all_runs_values, axis=0)
            std_values = np.std(all_runs_values, axis=0)

        label = labels[i] if labels else os.path.basename(dir_path)
        color = colors[i] if colors else plt.cm.rainbow(np.linspace(0, 1, len(json_paths)))[i]
        # Plot the average curve with shaded standard deviation region
        plt.plot(iterations, mean_values, label=label, linewidth=1.5, color=color)
        plt.fill_between(iterations, mean_values - std_values, mean_values + std_values, color=color, alpha=0.1)

    plt.legend(title='Algorithms', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title_fontsize='14')
    
    plt.ylabel(y_axis_name, fontsize=14, labelpad=10)
    plt.xlabel('Iterations', fontsize=14)
    plt.grid(True, linewidth=0.5)
    plt.yscale(vertical_axis_scale)
    ax = plt.gca()
    # Format y-axis labels in scientific notation
    #formatter = FuncFormatter(lambda y, _: '{:.1e}'.format(y))
    #ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.grid(which='minor', linestyle=':', linewidth=0.2)
    # Set the color of the edges of the main plot
    for spine in ax.spines.values():
        spine.set_edgecolor('darkgrey')
    # Set the color of the ticks
    ax.tick_params(axis='x', which='both', color='darkgrey')
    ax.tick_params(axis='y', which='both', color='darkgrey')
    plt.tight_layout()
    plt.savefig("figures/"+column_name+metrics_file+".png",dpi=400)


def plot_box_loss_from_json(json_paths, column_name, metrics_file="metrics", vertical_axis_scale="log", labels=None, colors=None):
    """
    Plot box plots for the loss values from JSON files over iterations.
    :param json_paths: list of paths to directories containing JSON files
    :param column_name: name of the column to plot
    """
    plt.clf()
    plt.figure(figsize=(6, 4))

    data_to_plot = []

    # Method's data directory
    for dir_path in json_paths:
        all_runs_data = []
        # One run's data directory
        for run_directory in os.listdir(dir_path):
            run_path = os.path.join(dir_path, run_directory)
            # Path to the metrics file
            if os.path.isdir(run_path):
                metrics_path = os.path.join(run_path, "metrics", metrics_file+".json")

                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r') as json_file:
                        data = [json.loads(line) for line in json_file]
                        if data == [] or all([elem is None for elem in data]):
                            continue
                        loss_values = [entry.get(column_name, None) for entry in data]
                        all_runs_data.extend(loss_values)

        # Skip if no data was found
        if all_runs_data == [] or all([elem is None for elem in all_runs_data]):
            continue

        data_to_plot.append(np.array(all_runs_data))

    labels = labels if labels else [os.path.basename(dir_path) for dir_path in json_paths]
    colors = colors if colors else plt.cm.rainbow(np.linspace(0, 1, len(json_paths)))
    # Create box plots
    box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.8)#, positions=[0.2, 1.1, 2.2, 3.1])

    # Customize box colors
    legend_handles = []
    for i, (box, color) in enumerate(zip(box_plot['boxes'], colors)):
        # Convert color to RGB
        color = matplotlib.colors.to_rgb(color)
        box.set_facecolor((*color, 0.2))  # Set the alpha of the facecolor to 0.2
        box.set_edgecolor((*color, 1))  # Set the alpha of the edgecolor to 1

        # Set the color of medians, whiskers, and caps to be the same as the box
        box_plot['medians'][i].set_color(color)
        box_plot['whiskers'][i*2].set_color(color)
        box_plot['whiskers'][i*2+1].set_color(color)
        box_plot['caps'][i*2].set_color(color)
        box_plot['caps'][i*2 + 1].set_color(color)
        
        # Set the color of the fliers
        for flier in box_plot['fliers'][i:i*2 + 1]:
            flier.set(marker='.', markerfacecolor=color, markersize=5, linestyle='none', markeredgecolor=color)
    
        # Create a patch for the box
        box_patch = Patch(facecolor=color, edgecolor=color, linewidth=1)
        box_patch.set_facecolor((color[0], color[1], color[2], 0.2))  # Set the alpha of the facecolor to 0.2
        box_patch.set_edgecolor((color[0], color[1], color[2], 1))  # Set the alpha of the edgecolor to 1
        median_line = Line2D([0], [0], color=color, alpha=1, lw=1)
        # Add the patch and lines to the legend handles
        legend_handles.append((box_patch, median_line))

    plt.legend(legend_handles, labels, handlelength=1, title='Algorithms', loc='center left', bbox_to_anchor=(1, 0.5), frameon=False, title_fontsize='14')

    def format_tick(value, tick_number):
        return f'{value:g}'

    plt.ylabel('Out-of-Sample MSE', fontsize=14, labelpad=10)
    plt.grid(True, linewidth=0.5)
    plt.yscale(vertical_axis_scale)
    ax = plt.gca()
    ax.set_yticks([1, 3, 10, 30])
    ax.set_yticklabels([1, 3, 10, 30])
    #plt.xticks([1, 2, 3, 4], [5000, 5000, 10000, 10000])
    ax.yaxis.set_major_formatter(FuncFormatter(format_tick))
    # Calculate the locations of the minor ticks
    minor_tick_locations = []
    for start, end in zip([1, 3, 10], [3, 10, 30]):
        minor_tick_locations.extend(np.arange(start, end, (end-start)/10))
    # Add minor grid lines at specific locations on the y-axis
    ax.yaxis.set_minor_locator(FixedLocator(minor_tick_locations))
    #ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(which='minor', linestyle=':', linewidth=0.2)
    # Set the color of the edges of the main plot
    for spine in ax.spines.values():
        spine.set_edgecolor('darkgrey')
    # Set the color of the ticks
    ax.tick_params(axis='x', which='both', color='darkgrey')
    ax.tick_params(axis='y', which='both', color='darkgrey')
    plt.tight_layout()
    plt.savefig("figures/"+column_name+metrics_file+".png", dpi=400)


# Setting colors, labels and paths to data
root = "/home/ipetruli/funcBO/applications/data/outputs/"
methods = ["dfiv5000", "funcBO_linear5000", "funcBO_dual_iterative5000", "parametricBO_BGS5000", "parametricBO_ITD5000"]
labels = ["DFIV", "FID linear", "FID", "CID", "ITD"]
custom_colors = ['#00bf7d', '#e76bf3', '#f8766d', '#00b0f6', '#bdb76b']#, '#d2c5b6']
json_paths = [root + item for item in methods]

# Plotting line plots
plot_line_loss_from_json(json_paths, "outer_loss", metrics_file="metrics", y_axis_name="Outer Loss", labels=labels, colors=custom_colors)
#plot_line_loss_from_json(json_paths, "loss", metrics_file="inner_metrics", y_axis_name="Inner Loss", labels=labels, colors=custom_colors)
#plot_line_loss_from_json(json_paths, "loss", vertical_axis_scale="linear", metrics_file="dual_metrics", y_axis_name="Dual Loss", labels=labels, colors=custom_colors)

# Plotting box plots
plot_box_loss_from_json(json_paths, "test loss", metrics_file="test_metrics", labels=labels, colors=custom_colors)
