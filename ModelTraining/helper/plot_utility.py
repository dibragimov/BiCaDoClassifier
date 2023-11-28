# draw and save graphs
import matplotlib.pyplot as plt
import os


def plot_and_save(history, save_dir, experiment_name, params):
    # Plot the model results
    if params is None:
        params = ['loss', 'accuracy']
    for p in params:
        par = history.history[p]
        val_par = history.history[f'val_{p}']
        epochs = range(len(par))
        plt.plot(epochs, par, 'r', label=f'Training {p}')
        plt.plot(epochs, val_par, 'b', label=f'Validation {p}')
        plt.title(f'Training and validation {p}')
        plt.legend()
        # plt.figure()
        plt.savefig(os.path.join(save_dir, f'{experiment_name}_{p}.jpg'))
    return
