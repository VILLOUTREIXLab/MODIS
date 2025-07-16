import pathlib

import matplotlib.pyplot as plt

from modis import load_log


def plot_training_log(
    training_mode: str,
    modality_names: list,
    checkpoint_path: pathlib.Path,
    use_best: bool = True,
    save_plot: bool = False,
    figsize: tuple = (15, 10)
) -> None:
    """Plot training log"""

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint path {checkpoint_path} doesn't exist.")

    if use_best:
        log_file = checkpoint_path / "checkpoint_log_best.json"
    else:
        log_file = checkpoint_path / "checkpoint_log_latest.json"
    log = load_log(log_file)

    num_modalities = len(log[0]['modal_recon_loss'])
    x = range(log[0]['epoch_idx'] + 1, log[-1]['epoch_idx'] + 2)

    # recon_loss = [item['recon_loss'] for item in log]
    modal_recon_loss = [[item['modal_recon_loss'][i] for item in log] for i in range(num_modalities)]
    # kl_loss = [item['kl_loss'] for item in log]
    kl_loss_modal = [[item['modal_kl_loss'][i] for item in log] for i in range(num_modalities)]
    d_train_loss = [item['d_train_loss'] for item in log]
    d_loss = [item['d_loss'] for item in log]
    g_loss = [item['g_loss'] for item in log]
    d_cluster_loss = [item['d_cluster_loss'] for item in log]
    d_aux_acc = [item['d_aux_acc'] for item in log]
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f'Training mode: {training_mode}', y=0.97)

    plt.subplot(2, 3, 1)
    # plt.plot(x, recon_loss, label='recon_loss')
    for i in range(num_modalities):
        plt.plot(x, modal_recon_loss[i], label='recon_loss_'+modality_names[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 2)
    # plt.plot(x, kl_loss, label='kl_loss')
    for i in range(num_modalities):
        plt.plot(x, kl_loss_modal[i], label='kl_loss_'+modality_names[i])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 3)
    plt.plot(x, d_train_loss, label='d_train_loss')
    plt.plot(x, d_loss, label='d_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(x, g_loss, label='g_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 5)
    plt.plot(x, d_cluster_loss, label='d_cluster_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 6)
    plt.plot(x, d_aux_acc, label='d_aux_acc')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add space for subtitle

    if save_plot:
        if use_best:
            figure_file = checkpoint_path / 'training_log_best.svg'
        else:
            figure_file = checkpoint_path / 'training_log_latest.svg'
        plt.savefig(
            figure_file,
            format='svg', 
            bbox_inches='tight'
        )
        plt.close()
    else:
        plt.show()
