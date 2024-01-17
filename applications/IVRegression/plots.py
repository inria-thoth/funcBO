import matplotlib.pyplot as plt
import numpy as np
import os
import json

def plot_line_loss_from_json(json_paths, column_name, metrics_file="metrics"):
    """
    Plot the average loss curve with standard deviation from JSON files over iterations.
    :param json_paths: list of paths to directories containing JSON files
    :param column_name: name of the column to plot
    """
    plt.clf()
    plt.figure(figsize=(8, 6))

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

        # Plot the average curve with shaded standard deviation region
        color = plt.cm.rainbow(np.linspace(0, 1, len(json_paths)))[i]
        label = os.path.basename(dir_path)
        plt.plot(iterations, mean_values, label=label, linewidth=2, color=color)
        plt.fill_between(iterations, mean_values - std_values, mean_values + std_values, color=color, alpha=0.1)

    plt.yscale('log')
    plt.xlabel('Iterations')
    plt.ylabel(column_name)
    plt.legend(loc='best')

    # Save the plot
    plt.tight_layout()
    plt.savefig("figures/" + column_name + metrics_file + ".png")


def plot_box_loss_from_json(json_paths, column_name, metrics_file="metrics"):
    """
    Plot box plots for the loss values from JSON files over iterations.
    :param json_paths: list of paths to directories containing JSON files
    :param column_name: name of the column to plot
    """
    plt.clf()
    plt.figure(figsize=(4, 3))

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

    # Create box plots
    labels = [os.path.basename(dir_path) for dir_path in json_paths]
    box_plot = plt.boxplot(data_to_plot, labels=labels, patch_artist=True, sym='', widths=0.7)

    # Customize box colors
    for i, box in enumerate(box_plot['boxes']):
        color = plt.cm.rainbow(np.linspace(0, 1, len(json_paths)))[i]
        box.set_edgecolor(color)
        box.set_alpha(0.2)
        box.set_facecolor(color)

        # Set the color of medians, whiskers, and caps to be the same as the box
        box_plot['medians'][i].set_color(color)
        box_plot['whiskers'][i*2].set_color(color)
        box_plot['whiskers'][i*2 + 1].set_color(color)
        box_plot['caps'][i*2].set_color(color)
        box_plot['caps'][i*2 + 1].set_color(color)

    plt.xlabel('Methods')
    plt.ylabel(column_name)
    plt.tight_layout()
    plt.grid(True, linewidth=0.5)
    plt.yticks([1, 10, 30])  # Set y-axis ticks
    plt.savefig("figures/" + column_name + metrics_file + ".png")

json_paths = ["/home/ipetruli/funcBO/applications/data/outputs/dfiv", "/home/ipetruli/funcBO/applications/data/outputs/funcBO", "/home/ipetruli/funcBO/applications/data/outputs/DFIV_with_norms", "/home/ipetruli/funcBO/applications/data/outputs/funcBO_inner_linear_composite_dual_linear_iterative", "/home/ipetruli/funcBO/applications/data/outputs/funcBO_inner_linear_iterative_dual_linear_closed"]
plot_line_loss_from_json(json_paths, "outer_loss", metrics_file="metrics")
plot_line_loss_from_json(json_paths, "val_loss", metrics_file="metrics")
plot_line_loss_from_json(json_paths, "test_loss", metrics_file="metrics")
plot_line_loss_from_json(json_paths, "loss", metrics_file="inner_metrics")
plot_line_loss_from_json(json_paths, "loss", metrics_file="dual_metrics")