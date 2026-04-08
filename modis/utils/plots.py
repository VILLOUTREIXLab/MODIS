"""
Plotting utilities for MODIS.

This module provides functions for visualising training logs, latent space
projections (2-D and 3-D PCA), confusion matrices, and cross-modal
reconstruction/translation heatmaps.
"""
import pathlib
import colorsys
from typing import Literal

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from modis import Model
from modis.utils.io import load_config
from modis.utils.data import get_dataloaders, get_samples_from_dataloader
from modis.utils.evaluation import calc_classification_metrics


def plot_training_log(
    training_mode: str,
    modality_names: list,
    checkpoint_dir: pathlib.Path,
    use_best: bool = True,
    save_plot: bool = False,
    figsize: tuple = (15, 10),
) -> None:
    """Plot training metrics from a saved training log.

    Reads a JSON log file from ``checkpoint_dir`` and produces a figure with
    subplots for reconstruction loss, KL divergence, discriminator losses,
    generator loss, gradient penalties (if applicable), and accuracy curves.

    Args:
        training_mode (str): Training mode string displayed in the figure
            title (e.g., ``'semisupervised'``).
        modality_names (list[str]): Display names for each modality.
        checkpoint_dir (pathlib.Path): Directory containing the training log
            JSON file.
        use_best (bool): If ``True``, reads ``checkpoint_log_best.json``;
            otherwise reads ``checkpoint_log_latest.json``.
            Defaults to ``True``.
        save_plot (bool): If ``True``, saves the figure as an SVG file in
            ``checkpoint_dir``; otherwise displays it interactively.
            Defaults to ``False``.
        figsize (tuple[int, int]): Figure dimensions ``(width, height)`` in
            inches. Defaults to ``(15, 10)``.

    Raises:
        FileNotFoundError: If ``checkpoint_dir`` or the log file does not
            exist.
    """
    from modis import load_log

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir {checkpoint_dir} doesn't exist.")

    log_file = checkpoint_dir / (
        "checkpoint_log_best.json" if use_best else "checkpoint_log_latest.json"
    )
    log = load_log(log_file)

    relativistic = len(log[0]['r']) > 0
    num_modalities = len(log[0]['modal_recon_loss'])
    x = range(log[0]['epoch_idx'] + 1, log[-1]['epoch_idx'] + 2)

    modal_recon_loss = [[item['modal_recon_loss'][i] for item in log] for i in range(num_modalities)]
    kl_loss_modal = [[item['modal_kl_loss'][i] for item in log] for i in range(num_modalities)]
    d_train_loss = [item['d_train_loss'] for item in log]
    d_loss = [item['d_loss'] for item in log]
    g_loss = [item['g_loss'] for item in log]
    if relativistic:
        penalties = [[item['r'][i] for item in log] for i in range(num_modalities)]
    d_adv_acc = [item['d_adv_acc'] for item in log]
    d_aux_acc = [item['d_aux_acc'] for item in log]
    if 'val_acc' in log[0]:
        val_acc = [item['val_acc'] for item in log]

    fig = plt.figure(figsize=figsize)
    fig.suptitle(f'Training mode: {training_mode}', y=0.97)

    plt.subplot(2, 3, 1)
    for i in range(num_modalities):
        plt.plot(x, modal_recon_loss[i], label=f'recon_loss_{modality_names[i]}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 2)
    for i in range(num_modalities):
        plt.plot(x, kl_loss_modal[i], label=f'kl_loss_{modality_names[i]}')
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

    if relativistic:
        plt.subplot(2, 3, 5)
        for i in range(num_modalities):
            plt.plot(x, penalties[i], label=f'R{i + 1}')
        plt.xlabel('epoch')
        plt.ylabel('penalty')
        plt.legend(prop={'size': 11})
        plt.grid()

    subplot_index = 6 if relativistic else 5
    plt.subplot(2, 3, subplot_index)
    plt.plot(x, d_adv_acc, label='d_adv_acc')
    plt.plot(x, d_aux_acc, label='d_aux_acc')
    if 'val_acc' in log[0]:
        plt.plot(x, val_acc, label='val_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_plot:
        suffix = 'best' if use_best else 'latest'
        figure_file = checkpoint_dir / f'training_log_{suffix}.svg'
        plt.savefig(figure_file, format='svg', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


colorblind_safe_colors = [
    '#E6194B', '#3CB44B', '#4363D8', '#42D4F4', '#F032E6',
    '#FFE119', '#000000', '#ADD8E6', '#F58231', '#911EB4',
    '#FABED4', '#A9A9A9',
]

colors = [
    "#FF0000", "#80FF00", "#0080C0", "#F58231",
    "#8000FF", "#FF80FF", "#FFFF00",
]


def get_colors(n: int = 5) -> list:
    """Return a list of ``n`` hexadecimal color codes.

    Falls back to a rainbow colormap when ``n`` exceeds the built-in palette.

    Args:
        n (int): Number of colors requested. Defaults to ``5``.

    Returns:
        list[str]: List of hexadecimal color strings of length ``n``.
    """
    if n > len(colors):
        import matplotlib.colors as mcolors
        colors_from_cmap = plt.cm.rainbow(np.linspace(0, 1, n))
        return [mcolors.rgb2hex(c) for c in colors_from_cmap]
    return colors[:n]


def generate_colors(hex_color: str, n: int) -> list:
    """Generate ``n`` color variants of a base hue by varying lightness and saturation.

    Args:
        hex_color (str): Base color in ``'#RRGGBB'`` hexadecimal format.
        n (int): Number of color variants to generate.

    Returns:
        list[str]: Generated colors in ``'#RRGGBB'`` hexadecimal format.
    """
    def hex_to_rgb(hex_str):
        return tuple(int(hex_str[i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    def rgb_to_hex(rgb):
        return f"#{int(rgb[0] * 255):02X}{int(rgb[1] * 255):02X}{int(rgb[2] * 255):02X}"

    r, g, b = hex_to_rgb(hex_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    step = 0.15
    start_point = 0.5 - (n * step / 2) - step
    if start_point < 0:
        step = 1 / (n + 2)
        start_point = step

    result = []
    for i in range(n):
        new_l = min(1, max(0, start_point + step + i * step))
        new_s = min(1, max(0, start_point + step + i * step))
        new_s = max(0.3, 1 + start_point - new_s)
        result.append(rgb_to_hex(colorsys.hls_to_rgb(h, new_l, new_s)))

    return result


def get_class_per_modality_colors(num_modalities: int, num_classes: int) -> list:
    """Return a flat list of colors for all class-modality combinations.

    Args:
        num_modalities (int): Number of modalities.
        num_classes (int): Number of classes.

    Returns:
        list[str]: Colors of length ``num_modalities * num_classes``, ordered
        by modality then class.
    """
    class_colors = get_colors(num_classes)
    colors_per_modality = list(zip(*[generate_colors(color, num_modalities) for color in class_colors]))
    return [color for modality_color in colors_per_modality for color in modality_color]


def plot_2d_projection(
    technique: Literal['pca', 'umap'],
    data,
    labels,
    labels_colors: list = None,
    labels_names: list = None,
    standardize: bool = True,
    checkpoint_dir: pathlib.Path = None,
    is_train: bool = None,
    is_best: bool = None,
    save_plot: bool = False,
    figsize: tuple = (8, 8),
    text_size: int = 20,
    alt_save_dir: pathlib.Path = None,
    alt_filename: str = None,
) -> None:
    """Display or save a 2-D dimensionality reduction scatter plot.

    Args:
        technique (str): Dimensionality reduction technique.
            Currently only ``'pca'`` is supported.
        data (array-like): Input data of shape ``(n_samples, n_features)``.
        labels (array-like): Integer class label per sample, shape
            ``(n_samples,)``.
        labels_colors (list[str], optional): Per-class hex color strings.
            Falls back to the built-in palette when ``None``.
        labels_names (list[str], optional): Human-readable class names.
            Falls back to ``'class {label}'`` when ``None``.
        standardize (bool): If ``True``, standardises ``data`` with
            :class:`~sklearn.preprocessing.StandardScaler` before reduction.
            Defaults to ``True``.
        checkpoint_dir (pathlib.Path, optional): Checkpoint directory used
            for building the default save path. Required when
            ``save_plot=True`` and ``alt_save_dir`` is ``None``.
        is_train (bool, optional): Whether the data is from the training set.
            Required when ``save_plot=True``.
        is_best (bool, optional): Whether the plot corresponds to the best
            checkpoint. Required when ``save_plot=True``.
        save_plot (bool): If ``True``, saves the figure as SVG; otherwise
            displays it interactively. Defaults to ``False``.
        figsize (tuple[int, int]): Figure size in inches. Defaults to
            ``(8, 8)``.
        text_size (int): Font size for axis labels and tick marks.
            Defaults to ``20``.
        alt_save_dir (pathlib.Path, optional): Alternative save directory that
            overrides the default path derived from ``checkpoint_dir``.
        alt_filename (str, optional): Alternative filename stem (without
            extension) that overrides the default naming convention.

    Raises:
        ValueError: If ``technique`` is not ``'pca'``, or if ``is_train`` or
            ``is_best`` are not booleans when ``save_plot=True``.
        FileNotFoundError: If ``checkpoint_dir`` does not exist when
            ``save_plot=True``.
    """
    if save_plot:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir {checkpoint_dir} doesn't exist.")
        if not isinstance(is_train, bool):
            raise ValueError("Parameter is_train must be boolean type")
        if not isinstance(is_best, bool):
            raise ValueError("Parameter is_best must be boolean type")

    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data

    if technique == 'pca':
        reducer = PCA(n_components=2)
        reduced_data = reducer.fit_transform(data_scaled)
        x_label, y_label = 'PC1', 'PC2'
    else:
        raise ValueError("Valid technique options are 'pca' and 'umap'")

    plt.figure(figsize=figsize)
    unique_labels = set(labels)
    for label in unique_labels:
        lc = labels_colors[label] if labels_colors else colors[label]
        named_label = labels_names[label] if labels_names else f'class {label}'
        mask = labels == label
        plt.scatter(reduced_data[mask, 0], reduced_data[mask, 1], c=lc, label=named_label, s=20)

    plt.xlabel(x_label, fontsize=text_size)
    plt.ylabel(y_label, fontsize=text_size)
    plt.xticks(fontsize=text_size)
    plt.yticks(fontsize=text_size)

    if len(unique_labels) < 10:
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), fontsize=text_size)

    if save_plot:
        if alt_save_dir is not None:
            filename = (
                f"{alt_filename}.svg" if alt_filename is not None
                else f"{technique}_2d_checkpoint_{'best' if is_best else 'latest'}.svg"
            )
            figure_file = alt_save_dir / filename
        else:
            split_name = 'train' if is_train else 'val'
            suffix = 'best' if is_best else 'latest'
            figure_file = checkpoint_dir / split_name / f"{technique}_2d_checkpoint_{suffix}.svg"
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_file, format='svg', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_3d_projection(
    technique: Literal['pca', 'umap'],
    data,
    labels,
    labels_colors: list = None,
    labels_names: list = None,
    standardize: bool = False,
    checkpoint_dir: pathlib.Path = None,
    is_train: bool = None,
    is_best: bool = True,
    save_plot: bool = False,
    width: int = 800,
    height: int = 800,
) -> None:
    """Display or save an interactive 3-D dimensionality reduction plot.

    Args:
        technique (str): Dimensionality reduction technique.
            Currently only ``'pca'`` is supported.
        data (array-like): Input data of shape ``(n_samples, n_features)``.
        labels (array-like): Integer class label per sample.
        labels_colors (list[str], optional): Per-class hex color strings.
        labels_names (list[str], optional): Human-readable class names.
        standardize (bool): If ``True``, standardises ``data`` before
            reduction. Defaults to ``False``.
        checkpoint_dir (pathlib.Path, optional): Checkpoint directory used
            for the default save path. Required when ``save_plot=True``.
        is_train (bool, optional): Whether data is from the training set.
            Required when ``save_plot=True``.
        is_best (bool): Whether the plot corresponds to the best checkpoint.
            Defaults to ``True``.
        save_plot (bool): If ``True``, saves the figure as SVG; otherwise
            displays it interactively. Defaults to ``False``.
        width (int): Plot width in pixels. Defaults to ``800``.
        height (int): Plot height in pixels. Defaults to ``800``.

    Raises:
        ValueError: If ``technique`` is not ``'pca'``, or if ``is_train`` or
            ``is_best`` are not booleans when ``save_plot=True``.
        FileNotFoundError: If ``checkpoint_dir`` does not exist when
            ``save_plot=True``.
    """
    if save_plot:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir {checkpoint_dir} doesn't exist.")
        if not isinstance(is_train, bool):
            raise ValueError("Parameter is_train must be boolean type")
        if not isinstance(is_best, bool):
            raise ValueError("Parameter is_best must be boolean type")

    if standardize:
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
    else:
        data_scaled = data

    if technique == 'pca':
        reducer = PCA(n_components=3)
        reduced_data = reducer.fit_transform(data_scaled)
        x_label, y_label, z_label, title = 'PC1', 'PC2', 'PC3', 'PCA'
    else:
        raise ValueError("technique must be either 'pca' or 'umap'")

    fig = go.Figure()
    for label in list(set(labels)):
        lc = labels_colors[label] if labels_colors else colors[label]
        named_label = labels_names[label] if labels_names else f'class {label}'
        mask = labels == label
        fig.add_trace(go.Scatter3d(
            x=reduced_data[mask, 0],
            y=reduced_data[mask, 1],
            z=reduced_data[mask, 2],
            mode='markers',
            marker=dict(size=2, color=lc, opacity=0.8),
            name=named_label,
        ))

    fig.update_layout(
        scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title=z_label),
        margin=dict(r=10, b=10, l=10, t=30),
        legend=dict(font=dict(size=14), itemsizing='constant'),
        title=title,
        width=width,
        height=height,
    )

    if save_plot:
        split_name = 'train' if is_train else 'val'
        suffix = 'best' if is_best else 'latest'
        figure_file = checkpoint_dir / split_name / f"{technique}_3d_checkpoint_{suffix}.svg"
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        pio.write_image(fig, figure_file)
    else:
        fig.show()


def plot_confusion_matrix(
    true_labels,
    pred_labels,
    performance_metrics: bool = False,
    checkpoint_dir: pathlib.Path = None,
    is_train: bool = None,
    is_best: bool = True,
    save_plot: bool = False,
    figsize: tuple = (16, 8),
    text_size: int = 20,
    filename_suffix: str = None,
    alt_save_dir: pathlib.Path = None,
) -> None:
    """Display or save a confusion matrix with optional performance metrics.

    Plots a colour-coded confusion matrix. When ``performance_metrics=True``,
    per-class precision and recall bars are appended alongside a text box
    containing aggregate metrics.

    Args:
        true_labels (array-like): Ground-truth labels, shape ``(n_samples,)``.
        pred_labels (array-like): Predicted labels, shape ``(n_samples,)``.
        performance_metrics (bool): If ``True``, includes precision, recall,
            and a metrics text box. Defaults to ``False``.
        checkpoint_dir (pathlib.Path, optional): Checkpoint directory used for
            the default save path. Required when ``save_plot=True`` and
            ``alt_save_dir`` is ``None``.
        is_train (bool, optional): Whether data is from the training set.
            Required when ``save_plot=True``.
        is_best (bool): Whether the plot corresponds to the best checkpoint.
            Defaults to ``True``.
        save_plot (bool): If ``True``, saves the figure as SVG; otherwise
            displays it. Defaults to ``False``.
        figsize (tuple[int, int]): Figure size in inches.
            Defaults to ``(16, 8)``.
        text_size (int): Font size for annotations. Defaults to ``20``.
        filename_suffix (str, optional): Suffix appended to the output
            filename (without leading underscore).
        alt_save_dir (pathlib.Path, optional): Alternative save directory that
            overrides the default path.

    Raises:
        ValueError: If ``is_train`` or ``is_best`` are not booleans when
            ``save_plot=True``.
        FileNotFoundError: If ``checkpoint_dir`` does not exist when
            ``save_plot=True``.
    """
    if save_plot:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir {checkpoint_dir} doesn't exist.")
        if not isinstance(is_train, bool):
            raise ValueError("Parameter is_train must be boolean type")
        if not isinstance(is_best, bool):
            raise ValueError("Parameter is_best must be boolean type")

    metrics = calc_classification_metrics(true_labels, pred_labels)
    metrics_text = (
        f"\n    ACC: {metrics['acc']:.3f}\n"
        f"    B-ACC: {metrics['bacc']:.3f}\n"
        f"    JI: {metrics['ji']:.3f}\n"
        f"    NMI: {metrics['nmi']:.3f}\n"
        f"    F1: {metrics['f1']:.3f}\n"
        f"    ARI: {metrics['ari']:.3f}\n    "
    )

    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    if performance_metrics:
        precision = np.nan_to_num(np.diag(cm) / np.where(cm.sum(axis=0) > 0, cm.sum(axis=0), 1e-9))
        recall = np.nan_to_num(np.diag(cm) / np.where(cm.sum(axis=1) > 0, cm.sum(axis=1), 1e-9))
        num_classes = len(np.unique(true_labels))

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[num_classes, 1, num_classes],
            height_ratios=[num_classes, 1],
            hspace=0.2,
            wspace=0.1,
        )

        disp.plot(ax=fig.add_subplot(gs[0, 0]), cmap=plt.cm.Blues,
                  text_kw={'fontsize': text_size}, colorbar=False)
        disp.ax_.set_aspect('equal')
        disp.ax_.tick_params(axis='both', which='major', labelsize=text_size)
        disp.ax_.xaxis.label.set_size(text_size)
        disp.ax_.yaxis.label.set_size(text_size)

        ax_precision = fig.add_subplot(gs[1, 0])
        ax_precision.imshow(precision.reshape(1, -1), cmap=plt.cm.Blues, aspect='equal', vmin=0, vmax=1)
        for j, v in enumerate(precision):
            ax_precision.text(j, 0, f'{v:.2f}', ha='center', va='center',
                              fontsize=text_size, color='white' if v > 0.7 else 'black')
        ax_precision.set_xticks([])
        ax_precision.set_yticks([])
        ax_precision.set_xlabel('Precision', fontsize=text_size)

        ax_recall = fig.add_subplot(gs[0, 1])
        ax_recall.imshow(recall.reshape(-1, 1), cmap=plt.cm.Blues, aspect='equal', vmin=0, vmax=1)
        for i, v in enumerate(recall):
            ax_recall.text(0, i, f'{v:.2f}', ha='center', va='center',
                           fontsize=text_size, color='white' if v > 0.7 else 'black')
        ax_recall.set_yticks([])
        ax_recall.set_xticks([])
        ax_recall.set_xlabel('Recall', fontsize=text_size)

        ax_metrics = fig.add_subplot(gs[0:2, 2])
        ax_metrics.text(0.5, 0.5, metrics_text, fontsize=text_size, color='black',
                        ha='right', va='center', wrap=True)
        ax_metrics.set_xticks([])
        ax_metrics.set_yticks([])
        ax_metrics.set_frame_on(False)
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': (4, 1)})
        disp.plot(ax=axs[0], cmap=plt.cm.Blues, text_kw={'fontsize': text_size}, colorbar=False)
        axs[0].xaxis.label.set_size(text_size)
        axs[0].yaxis.label.set_size(text_size)
        axs[0].tick_params(axis='both', which='major', labelsize=12)
        axs[1].text(0.5, 0.5, metrics_text, fontsize=text_size, color='black',
                    ha='right', va='center', wrap=True)
        axs[1].set_axis_off()
        plt.tight_layout()

    if save_plot:
        save_dir = alt_save_dir if alt_save_dir is not None else (
            checkpoint_dir / ('train' if is_train else 'val')
        )
        suffix = f"_{filename_suffix}" if filename_suffix else ''
        chk = 'best' if is_best else 'latest'
        figure_file = save_dir / f"confusion_matrix_checkpoint_{chk}{suffix}.svg"
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_file, format='svg', bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def checkpoint_report_plots(
    checkpoint_dir: pathlib.Path,
    datasets: list,
    is_train: bool,
    use_best: bool = True,
    num_samples: int = None,
) -> None:
    """Generate and save a full set of diagnostic plots for a checkpoint.

    Saves a training-log curve, 2-D and 3-D PCA projections of the latent
    space, and confusion matrices (overall and per-modality) to
    ``checkpoint_dir``.

    Args:
        checkpoint_dir (pathlib.Path): Directory containing the checkpoint
            file and ``config.yaml``.
        datasets (list[torch.utils.data.Dataset]): Source datasets, one per
            modality.
        is_train (bool): If ``True``, plots are tagged as training-set
            results; otherwise as validation-set results.
        use_best (bool): If ``True``, loads ``checkpoint_best.pth``;
            otherwise loads ``checkpoint_latest.pth``.
            Defaults to ``True``.
        num_samples (int, optional): Maximum number of samples to use per
            modality. If ``None``, all samples are used.

    Raises:
        NotADirectoryError: If ``checkpoint_dir`` does not exist.
        FileNotFoundError: If the specified checkpoint file does not exist.
    """
    config = load_config(checkpoint_dir / 'config.yaml')
    chk_name = 'checkpoint_best.pth' if use_best else 'checkpoint_latest.pth'
    checkpoint_file = checkpoint_dir / chk_name
    num_modalities = len(config.modalities)
    modality_names = [m.name for m in config.modalities]

    if not checkpoint_dir.exists():
        raise NotADirectoryError("Checkpoint path doesn't exist")
    if not checkpoint_file.exists():
        raise FileNotFoundError("The specified checkpoint file does not exist.")

    plot_training_log(
        training_mode=config.training_mode,
        modality_names=modality_names,
        checkpoint_dir=checkpoint_dir,
        use_best=use_best,
        save_plot=True,
    )

    model = Model(config)
    model.load_from_checkpoint(checkpoint_file, verbose=False)

    dataloaders = get_dataloaders(datasets, batch_size=config.batch_size, drop_last=False, shuffle=True)
    x, y = list(zip(*[
        get_samples_from_dataloader(dl, num_samples=num_samples, device=config.device)
        for dl in dataloaders
    ]))

    if config.training_mode == 'semisupervised':
        x, y = list(x), list(y)
        for i in range(num_modalities):
            labeled_mask = torch.tensor([label != -1 for label in y[i]])
            x[i] = x[i][labeled_mask]
            y[i] = y[i][labeled_mask]

    if sum(len(ds) for ds in y) == 0:
        print("At least some labeled samples are needed for plotting.")
        return

    modal_latents = [model.get_latents(x[i], input_modality=i) for i in range(num_modalities)]
    latents = torch.concat(modal_latents, dim=0).cpu().numpy()

    class_labels = torch.concat(y, dim=0).cpu().numpy()
    modality_labels = torch.concat(
        [torch.full((x[i].shape[0],), i) for i in range(num_modalities)], dim=0
    ).numpy()
    class_per_modality_labels = [
        cl + mi * config.num_classes
        for mi, modality_class_labels in enumerate(y)
        for cl in modality_class_labels.cpu().numpy()
    ]

    class_per_modality_colors = get_class_per_modality_colors(
        num_modalities=num_modalities, num_classes=config.num_classes
    )
    class_per_modality_names = [
        f"{config.modalities[mi].name}_{ci}"
        for mi in range(num_modalities)
        for ci in range(config.num_classes)
    ]

    plot_2d_projection(
        technique='pca',
        data=latents,
        labels=class_per_modality_labels,
        labels_colors=class_per_modality_colors,
        labels_names=class_per_modality_names,
        standardize=False,
        checkpoint_dir=checkpoint_dir,
        is_train=is_train,
        is_best=use_best,
        save_plot=True,
        text_size=26,
    )

    plot_3d_projection(
        technique='pca',
        data=latents,
        labels=class_per_modality_labels,
        labels_colors=class_per_modality_colors,
        labels_names=class_per_modality_names,
        standardize=False,
        checkpoint_dir=checkpoint_dir,
        is_train=is_train,
        is_best=use_best,
        save_plot=True,
    )

    class_pred, modality_pred = model.discriminator.predict(
        torch.tensor(latents).to(config.device), include_modality_pred=True
    )
    class_pred = class_pred.cpu().numpy()
    modality_pred = modality_pred.cpu().numpy()
    class_per_modality_labels_pred = np.array([
        cp + mp * config.num_classes
        for cp, mp in zip(class_pred, modality_pred)
    ])

    plot_confusion_matrix(
        class_labels, class_pred,
        performance_metrics=True,
        checkpoint_dir=checkpoint_dir,
        is_train=is_train, is_best=use_best,
        save_plot=True, figsize=(16, 10), text_size=25,
    )

    plot_confusion_matrix(
        class_per_modality_labels, class_per_modality_labels_pred,
        performance_metrics=True,
        checkpoint_dir=checkpoint_dir,
        is_train=is_train, is_best=use_best,
        save_plot=True, figsize=(40, 20), text_size=25,
        filename_suffix='class_per_modality',
    )

    for i in range(num_modalities):
        plot_confusion_matrix(
            class_labels[modality_labels == i],
            class_pred[modality_labels == i],
            performance_metrics=True,
            checkpoint_dir=checkpoint_dir,
            is_train=is_train, is_best=use_best,
            save_plot=True, figsize=(16, 10), text_size=25,
            filename_suffix=f'modality_{modality_names[i]}',
        )


def calc_reconstruction_and_translation_mse(
    x: list,
    model,
    save_plot: bool = False,
    save_dir: pathlib.Path = pathlib.Path('.'),
    suffix: str = None,
    text_size: int = 30,
    vmin: float = 0,
    vmax: float = 0.1,
) -> None:
    """Compute and plot a heatmap of reconstruction and translation MSE.

    For each pair of input and output modalities, computes the mean squared
    error between the original data and the model's reconstructed or
    translated version, then displays the results as a colour-coded heatmap.

    Args:
        x (list[torch.Tensor]): Per-modality input tensors.
        model: The trained :class:`~modis.nn.Model`.
        save_plot (bool): If ``True``, saves the figure as SVG; otherwise
            displays it. Defaults to ``False``.
        save_dir (pathlib.Path): Directory to save the figure when
            ``save_plot=True``. Defaults to the current directory.
        suffix (str, optional): Suffix appended to the output filename.
        text_size (int): Font size for annotations. Defaults to ``30``.
        vmin (float): Minimum value for the heatmap colour scale.
            Defaults to ``0``.
        vmax (float): Maximum value for the heatmap colour scale.
            Defaults to ``0.1``.

    Raises:
        FileNotFoundError: If ``save_dir`` does not exist when
            ``save_plot=True``.
    """
    if save_plot and not save_dir.exists():
        raise FileNotFoundError(f"Save directory {save_dir} doesn't exist.")

    num_modalities = len(x)

    translations = {}
    for input_modality in range(num_modalities):
        translations[input_modality] = {}
        for output_modality in range(num_modalities):
            translations[input_modality][output_modality] = model.translate(
                x[input_modality],
                input_modality=input_modality,
                output_modality=output_modality,
            )

    mse_matrix = np.zeros((num_modalities, num_modalities))
    for ii in range(num_modalities):
        for it in range(num_modalities):
            mse_matrix[ii, it] = np.mean(
                (translations[ii][it] - x[it].cpu().numpy()) ** 2
            )

    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(
        mse_matrix, annot=True, fmt=".3f", cmap="coolwarm",
        linewidths=0.5, cbar=True, annot_kws={"size": text_size},
        vmin=vmin, vmax=vmax,
    )

    plt.xlabel("output modality", fontsize=text_size)
    plt.ylabel("input modality", fontsize=text_size)
    ticks = [str(i + 1) for i in range(num_modalities)]
    ax.set_xticklabels(ticks, fontsize=text_size)
    ax.set_yticklabels(ticks, fontsize=text_size)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=text_size)

    if save_plot:
        fname = f"reconstruction_translation_mse_matrix{suffix if suffix else ''}.svg"
        plt.savefig(save_dir / fname, format='svg', bbox_inches='tight')
        plt.close()
    else:
        plt.show()