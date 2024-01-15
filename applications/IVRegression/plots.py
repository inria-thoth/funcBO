from funcBO.utils import plot_loss_from_json

json_paths = ["/home/ipetruli/funcBO/applications/data/outputs/dfiv/1/metrics/metrics.json", "/home/ipetruli/funcBO/applications/data/outputs/funcBO/1/metrics/metrics.json"]
plot_loss_from_json(json_paths, "outer_loss")
plot_loss_from_json(json_paths, "test_loss")