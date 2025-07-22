import pathlib
import colorsys
from typing import Literal

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# import umap.umap_ as umap
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from modis.model import MODIS
from modis.utils.config import load_config
from modis.utils.data import get_dataloaders, get_samples_from_dataloader
from modis.utils.utils import calc_classification_metrics


def plot_training_log(
    training_mode: str,
    modality_names: list,
    checkpoint_dir: pathlib.Path,
    use_best: bool = True,
    save_plot: bool = False,
    figsize: tuple = (15, 10)
) -> None:
    """Plot training log"""
    from modis import load_log  # Import here to avoid partially initialized module error 

    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir {checkpoint_dir} doesn't exist.")

    if use_best:
        log_file = checkpoint_dir / "checkpoint_log_best.json"
    else:
        log_file = checkpoint_dir / "checkpoint_log_latest.json"
    log = load_log(log_file)

    relativistic = len(log[0]['r']) > 0
    num_modalities = len(log[0]['modal_recon_loss'])
    x = range(log[0]['epoch_idx'] + 1, log[-1]['epoch_idx'] + 2)

    # recon_loss = [item['recon_loss'] for item in log]
    modal_recon_loss = [[item['modal_recon_loss'][i] for item in log] for i in range(num_modalities)]
    # kl_loss = [item['kl_loss'] for item in log]
    kl_loss_modal = [[item['modal_kl_loss'][i] for item in log] for i in range(num_modalities)]
    d_train_loss = [item['d_train_loss'] for item in log]
    d_loss = [item['d_loss'] for item in log]
    g_loss = [item['g_loss'] for item in log]
    # d_cluster_loss = [item['d_cluster_loss'] for item in log]
    if relativistic:
        penalties = [[item['r'][i] for item in log] for i in range(num_modalities)]
    d_adv_acc = [item['d_adv_acc'] for item in log]
    d_aux_acc = [item['d_aux_acc'] for item in log]
    if 'val_acc' in log[0]:
        val_acc = [item['val_acc'] for item in log]
    
    fig = plt.figure(figsize=figsize)
    fig.suptitle(f'Training mode: {training_mode}', y=0.97)

    plt.subplot(2, 3, 1)
    # plt.plot(x, recon_loss, label='recon_loss')
    for i in range(num_modalities):
        plt.plot(x, modal_recon_loss[i], label=f'recon_loss_{modality_names[i]}')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    plt.subplot(2, 3, 2)
    # plt.plot(x, kl_loss, label='kl_loss')
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

    # plt.subplot(2, 3, 4)
    # plt.plot(x, d_cluster_loss, label='d_cluster_loss')
    # plt.xlabel('epoch')
    # plt.ylabel('loss')
    # plt.legend(prop={'size': 11})
    # plt.grid()

    plt.subplot(2, 3, 4)
    plt.plot(x, g_loss, label='g_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(prop={'size': 11})
    plt.grid()

    if relativistic:
        plt.subplot(2, 3, 5)
        for i in range(num_modalities):
            plt.plot(x, penalties[i], label=f'R{i+1}')
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

    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Add space for subtitle

    if save_plot:
        if use_best:
            figure_file = checkpoint_dir / 'training_log_best.svg'
        else:
            figure_file = checkpoint_dir / 'training_log_latest.svg'
        plt.savefig(
            figure_file,
            format='svg', 
            bbox_inches='tight'
        )
        plt.close()
    else:
        plt.show()


colorblind_safe_colors = [
    '#E6194B',
    '#3CB44B',
    '#4363D8',
    '#42D4F4',
    '#F032E6',
    '#FFE119',
    '#000000',
    '#ADD8E6',
    '#F58231',
    '#911EB4',
    '#FABED4',
    '#A9A9A9'
]

colors = [
    "#FF0000",   # Red
    "#80FF00",   # Lime
    "#0080C0",   # Medium Blue    
    "#F58231",   # Orange
    "#8000FF",   # Indigo
    "#FF80FF",   # Lavender
    "#FFFF00",   # Yellow
    #"#00FFFF",   # Cyan
]

def get_colors(n=5) -> list[str]:
    """Return a list of hexadecimal color codes """
    # assert n <= len(colors), f"Only {len(colors)} available"
    if n > len(colors):
        import matplotlib.colors as mcolors
        colors_from_cmap = plt.cm.rainbow(np.linspace(0, 1, n))
        hex_colors = [mcolors.rgb2hex(c) for c in colors_from_cmap]
        return hex_colors
    return colors[:n]

def generate_colors(hex_color:str, n:int):
    """
    Generate n versions of the hue in hex_color by changing the lightness and saturation

    Args:
        hex_color (str): Color in hex format
        n (int): Number of colors to generate

    Return:
        (list): List of generated colors
    """
    def hex_to_rgb(hex):
        return tuple(int(hex[i:i+2], 16) / 255.0 for i in (1, 3, 5))

    def rgb_to_hex(rgb):
        return f"#{int(rgb[0]*255):02X}{int(rgb[1]*255):02X}{int(rgb[2]*255):02X}"

    r, g, b = hex_to_rgb(hex_color)
    
    # Convert RGB to HSL
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    step = 0.15
    start_point = 0.5 - (n*step/2)  - step
    if start_point < 0:
        step = 1/(n+2)
        start_point = step
    
    # Generate colors
    colors = []
    for i in range(n):
        # Adjust lightness
        new_l = min(1, max(0, start_point + step + i*step))
        
        # Adjust saturation
        new_s = min(1, max(0, start_point + step + i*step))
        new_s = 1 + start_point - new_s
        new_s = max(0.3, new_s)
        
        # Convert back to RGB and then to hex
        new_rgb = colorsys.hls_to_rgb(h, new_l, new_s)
        colors.append(rgb_to_hex(new_rgb))
    
    return colors

def get_class_per_modality_colors(num_modalities, num_classes):
    """Return a list of colors per modality per class"""
    class_colors = get_colors(num_classes)
    colors_per_modality = list(zip(*[generate_colors(color, num_modalities) for color in class_colors]))
    colors = [color for modality_color in colors_per_modality for color in modality_color]
    return colors

def plot_2d_projection(
    technique: Literal['pca'] | Literal['umap'],
    data,
    labels,
    labels_colors:  list[str] | None = None,
    labels_names: list[str] | None = None,
    standardize: bool = True,
    checkpoint_dir: pathlib.Path = None,
    is_train: bool | None = None,
    is_best: bool | None = None,
    save_plot: bool = False,
    figsize: tuple = (8, 8),
    text_size: int = 20
) -> None:
    """
    Display and save a 2D PCA or UMAP plot
    
    Args:
        technique (str): Choose between 'pca' or 'umap' dimensionality reduction technique
        labels_colors (None | list[str]): One hex color code for each unique class label
        labels_names (None | list[str]): One name for each unique class label
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
        x_label = 'PC1'
        y_label = 'PC2'
        title = 'PCA'
    # elif technique == 'umap':  ### Requires previous version of numpy and causes other mild compatibility issues
    #     reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2)
    #     reduced_data = reducer.fit_transform(data_scaled)
    #     x_label = 'UMAP component 1'
    #     y_label = 'UMAP component 2'
    #     title = 'UMAP'
    else:
        raise ValueError("Valid technique options are 'pca' and 'umap'")

    plt.figure(figsize=figsize)
    
    # Iterate through unique labels
    unique_labels = set(labels)
    for label in unique_labels:
        # Determine color
        if labels_colors:
            lc = labels_colors[label]
        else:
            lc = colors[label]
        
        # Determine label name
        if labels_names:
            named_label = labels_names[label]
        else:
            named_label = f'class {label}'
        
        # Create mask for current label
        mask = labels == label
        
        plt.scatter(
            reduced_data[mask, 0], 
            reduced_data[mask, 1], 
            c=lc, 
            label=named_label, 
            s=20
        )

    plt.xlabel(x_label, fontsize=text_size)
    plt.ylabel(y_label, fontsize=text_size)
    plt.xticks(fontsize=text_size)
    plt.yticks(fontsize=text_size)
    # plt.title(title, fontsize=text_size)

    if len(unique_labels) < 10:  ### Find better parameter and use also for 3d plots
        plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1), fontsize=text_size)  # bbox_to_anchor=(1.6, 1)

    if save_plot:
        split_name = 'train' if is_train else 'val'
        if is_best:
            figure_file = checkpoint_dir / split_name / f"{technique}_2d_checkpoint_best.svg"
        else:
            figure_file = checkpoint_dir / split_name / f"{technique}_2d_checkpoint_latest.svg"
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_file, format='svg', bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_3d_projection(
    technique: Literal['pca', 'umap'],
    data,
    labels,
    labels_colors:  list[str] | None = None,
    labels_names: list[str] | None = None,
    standardize: bool = False,
    checkpoint_dir: pathlib.Path = None,
    is_train: bool | None = None,
    is_best: bool = True,
    save_plot: bool = False,
    width: int = 800,
    height: int = 800
) -> None:
    """
    Display and save a 3D dimensionality reduction plot
    
    Args:
        technique (str): Choose between 'pca' or 'umap' dimensionality reduction technique
        labels_colors (None | list[str]): List of colors for each unique label
        labels_names (None | list[str]): List of names for each unique label
        standardize (boolean): Whether to standardize the data before reduction
        width (int): Width of the plot
        height (int): Height of the plot
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
        x_label = 'PC1'
        y_label = 'PC2'
        z_label = 'PC3'
        title = 'PCA'
    # elif technique == 'umap':
    #     reducer = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=3)
    #     reduced_data = reducer.fit_transform(data_scaled)
    #     x_label = 'UMAP 1'
    #     y_label = 'UMAP 2'
    #     z_label = 'UMAP 3'
    #     title = 'UMAP'
    else:
        raise ValueError("technique must be either 'pca' or 'umap'")

    fig = go.Figure()

    # Iterate through unique labels
    unique_labels = list(set(labels))
    for label in unique_labels:
        # Determine color
        if labels_colors:
            lc = labels_colors[label]
        else:
            lc = colors[label]
        
        # Determine label name
        if labels_names:
            named_label = labels_names[label]
        else:
            named_label = f'class {label}'
        
        # Create mask for current label
        mask = labels == label
        
        # Add trace for current label
        fig.add_trace(go.Scatter3d(
            x=reduced_data[mask, 0],
            y=reduced_data[mask, 1],
            z=reduced_data[mask, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=lc,
                opacity=0.8,
            ),
            name=named_label
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=z_label
        ),
        margin=dict(r=10, b=10, l=10, t=30),
        legend=dict(
            font=dict(size=14),
            itemsizing='constant'
        ),
        title=title,
        width=width,
        height=height
    )

    if save_plot:
        split_name = 'train' if is_train else 'val'
        if is_best:
            figure_file = checkpoint_dir / split_name / f"{technique}_3d_checkpoint_best.svg"
        else:
            figure_file = checkpoint_dir / split_name / f"{technique}_3d_checkpoint_latest.svg"
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        pio.write_image(fig, figure_file)
    else:
        fig.show()

def plot_confusion_matrix(
    true_labels,
    pred_labels,
    performance_metrics: bool = False,
    checkpoint_dir: pathlib.Path = None,
    is_train: bool | None = None,
    is_best: bool = True,
    save_plot: bool = False,
    figsize = (16, 8),
    text_size = 20,
    filename_suffix: str | None = None
) -> None:
    """
    Display a confusion matrix.

    Note: If there are many classes adjust figsize to fix recall and precision plots, add n to each
    
    Args:
        true_labels (array-like of shape (n_samples,))
        pred_labels (array-like of shape (n_samples,))
        performance_metrics: if True, plot recall and precision plots
    """
    if save_plot:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir {checkpoint_dir} doesn't exist.")
        if not isinstance(is_train, bool):
            raise ValueError("Parameter is_train must be boolean type")
        if not isinstance(is_best, bool):
            raise ValueError("Parameter is_best must be boolean type")

    metrics = calc_classification_metrics(true_labels, pred_labels)
    matrics_text = f"""
    ACC: {metrics['acc']:.3f}
    IFW-AAC: {metrics['inv-freq-w-acc']:.3f}
    B-AAC: {metrics['bacc']:.3f}
    JI: {metrics['ji']:.3f}
    NMI: {metrics['nmi']:.3f}
    F1: {metrics['f1']:.3f}
    ARI: {metrics['ari']:.3f}
    """

    cm = confusion_matrix(true_labels, pred_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm) #, display_labels=['num'+str(n) for n in range(10)])
    
    if performance_metrics:
        # Calculate precision and recall
        precision = np.nan_to_num(np.diag(cm) / np.where(cm.sum(axis=0) > 0, cm.sum(axis=0), 1e-9))
        recall = np.nan_to_num(np.diag(cm) / np.where(cm.sum(axis=1) > 0, cm.sum(axis=1), 1e-9))

        num_classes = len(np.unique(true_labels))

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(
            2, 3,
            width_ratios=[num_classes, 1, num_classes],  # (1-(1/num_classes))
            height_ratios=[num_classes, 1],
            hspace = 0.2,
            wspace = 0.1
        )
    
        disp.plot(ax=fig.add_subplot(gs[0, 0]), cmap=plt.cm.Blues, text_kw={'fontsize': text_size}, colorbar=False)
        disp.ax_.set_aspect('equal')
        # disp.ax_.set_title('Confusion Matrix', fontsize=text_size)
        disp.ax_.tick_params(axis='both', which='major', labelsize=text_size)  # Increase tick label size
        disp.ax_.xaxis.label.set_size(text_size)
        disp.ax_.yaxis.label.set_size(text_size)
    
        # Add precision plot
        ax_precision = fig.add_subplot(gs[1, 0])
        precision_2d = precision.reshape(1, -1)
        ax_precision.imshow(precision_2d, cmap=plt.cm.Blues, aspect='equal', vmin=0, vmax=1)
        for j, v in enumerate(precision):
            ax_precision.text(j, 0, f'{v:.2f}', ha='center', va='center', fontsize=text_size, color='white' if v > 0.7 else 'black')  # plt.cm.Blues(1)
        ax_precision.set_xticks([])
        ax_precision.set_yticks([])
        ax_precision.set_xlabel('Precision', fontsize=text_size)

        # Add recall plot
        ax_recall = fig.add_subplot(gs[0, 1])
        recall_2d = recall.reshape(-1, 1)
        ax_recall.imshow(recall_2d, cmap=plt.cm.Blues, aspect='equal', vmin=0, vmax=1)
        for i, v in enumerate(recall):
            ax_recall.text(0, i, f'{v:.2f}', ha='center', va='center', fontsize=text_size, color='white' if v > 0.7 else 'black')
        ax_recall.set_yticks([])
        ax_recall.set_xticks([])
        ax_recall.set_xlabel('Recall', fontsize=text_size)
    
        # Metrics
        ax_metrics = fig.add_subplot(gs[0:2, 2])
        ax_metrics.text(0.5, 0.5, matrics_text, fontsize=text_size, color='black', ha='right', va='center', wrap=True)
        ax_metrics.set_xticks([])
        ax_metrics.set_yticks([])
        ax_metrics.set_frame_on(False)
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios': (4,1)})
        disp.plot(ax=axs[0], cmap=plt.cm.Blues, text_kw={'fontsize': text_size}, colorbar=False)
        # axs[0].set_title('Confusion Matrix', fontsize=text_size)
        axs[0].xaxis.label.set_size(text_size)
        axs[0].yaxis.label.set_size(text_size)
        axs[0].tick_params(axis='both', which='major', labelsize=12)
        axs[1].text(0.5, 0.5, matrics_text, fontsize=text_size, color='black', ha='right', va='center', wrap=True)
        axs[1].set_axis_off()
    
        plt.tight_layout()

    if save_plot:
        split_name = 'train' if is_train else 'val'
        if filename_suffix is None:
            suffix = ''
        else:
            suffix = f"_{filename_suffix}"
        if is_best:
            figure_file = checkpoint_dir / split_name / f"confusion_matrix_checkpoint_best{suffix}.svg"
        else:
            figure_file = checkpoint_dir / split_name / f"confusion_matrix_checkpoint_latest{suffix}.svg"
        figure_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(figure_file, format = 'svg', bbox_inches = 'tight')  ### dpi=300
        plt.close()
    else:
        plt.show()

def checkpoint_report_plots(
    checkpoint_dir: pathlib.Path,
    config_file: pathlib.Path,
    datasets: list[torch.utils.data.Dataset],
    is_train: bool,
    use_best: bool = True,
    num_samples: int | None = None
) -> None:
    """Save checkpoint log, 2D pca, and confusion matrices"""
    config = load_config(config_file)
    checkpoint_file = checkpoint_dir / f"{'checkpoint_best.pth' if use_best else 'checkpoint_latest.pth'}"
    num_modalities = len(config.modalities)
    modality_names = [m.name for m in config.modalities]

    if not checkpoint_dir.exists():
        raise NotADirectoryError("Checkpoint path doesn't exist")

    if not checkpoint_file.exists():
        raise FileNotFoundError("There is not a best checkpoint file on this path.")

    plot_training_log(
        training_mode = config.training_mode,
        modality_names = modality_names,
        checkpoint_dir = checkpoint_dir,
        use_best = use_best,
        save_plot = True
    )

    model = MODIS(config)
    model.load_from_checkpoint(checkpoint_file, verbose=False)

    dataloaders = get_dataloaders(datasets, batch_size=config.batch_size, drop_last=False, shuffle=True)

    # Prepare required variables

    x, y = list(zip(*[get_samples_from_dataloader(dataloader, num_samples=num_samples, device=config.device) for dataloader in dataloaders]))

    # Remove unlabeled samples
    if config.training_mode == 'semisupervised':
        x = list(x)
        y = list(y)
        for i in range(num_modalities):
            labeled_mask = torch.tensor([True if label != -1 else False for label in y[i]])
            x[i] = x[i][labeled_mask]
            y[i] = y[i][labeled_mask]

    total_labeled_samples = sum([len(ds) for ds in y])
    if total_labeled_samples == 0:
        print("At least some labeled samples are needed for plotting the projections and confusion matrices")
        return

    # Latents
    modal_latents = [model.get_latents(x[i], input_modality=i) for i in range(num_modalities)]
    latents = torch.concat(modal_latents, dim=0).cpu().numpy()

    # Labels
    class_labels = torch.concat(y, dim=0).cpu().numpy()
    modality_labels = torch.concat([torch.full((x[i].shape[0],), i) for i in range(num_modalities)], dim=0).numpy()
    class_per_modality_labels = [cl + mi*config.num_classes for mi, modality_class_labels in enumerate(y) for cl in modality_class_labels.cpu().numpy()]  ####### Assuming equal number of classes per modality

    # Colors
    # modality_colors = get_class_per_modality_colors(num_modalities=1, num_classes=num_modalities)
    # class_colors = get_class_per_modality_colors(num_modalities=1, num_classes=config.num_classes)
    class_per_modality_colors = get_class_per_modality_colors(num_modalities=num_modalities, num_classes=config.num_classes)

    # Names
    modality_names = [f"{config.modalities[i].name}" for i in range(num_modalities)]
    # class_names = [i for i in range(config.num_classes)]
    class_per_modality_names = [f"{config.modalities[mi].name}_{ci}" for mi in range(num_modalities) for ci in range(config.num_classes)]

    # Plot projections

    plot_2d_projection(
        technique = 'pca',
        data = latents,
        labels = class_per_modality_labels,
        labels_colors = class_per_modality_colors,
        labels_names = class_per_modality_names,
        standardize = False,
        checkpoint_dir = checkpoint_dir,
        is_train = is_train,
        is_best = use_best,
        save_plot = True,
        text_size = 26
    )

    plot_3d_projection(
        technique = 'pca',
        data = latents,
        labels = class_per_modality_labels,
        labels_colors = class_per_modality_colors,
        labels_names = class_per_modality_names,
        standardize = False,
        checkpoint_dir = checkpoint_dir,
        is_train = is_train,
        is_best = use_best,
        save_plot = True
    )

    # Plot confusion matrices

    class_pred, modality_pred = model.discriminator.predict(torch.tensor(latents).to(config.device), include_modality_pred=True)#.cpu().numpy()
    class_pred = class_pred.cpu().numpy()
    modality_pred = modality_pred.cpu().numpy()
    class_per_modality_labels_pred = np.array([
        class_pred + modality * config.num_classes
        for class_pred, modality in zip(class_pred, modality_pred)
    ])

    plot_confusion_matrix(
        class_labels,
        class_pred,
        performance_metrics = True,
        checkpoint_dir = checkpoint_dir,
        is_train = is_train,
        is_best = use_best,
        save_plot = True,
        figsize = (16, 10),
        text_size = 25
    )

    # Class per modality
    plot_confusion_matrix(
        class_per_modality_labels,
        class_per_modality_labels_pred,
        performance_metrics = True,
        checkpoint_dir = checkpoint_dir,
        is_train = is_train,
        is_best = use_best,
        save_plot = True,
        figsize = (40, 20),
        text_size = 25,
        filename_suffix = 'class_per_modality'
    )

    for i in range(num_modalities):
        plot_confusion_matrix(
            class_labels[modality_labels == i],
            class_pred[modality_labels == i],
            performance_metrics = True,
            checkpoint_dir = checkpoint_dir,
            is_train = is_train,
            is_best = use_best,
            save_plot = True,
            figsize = (16, 10),
            text_size = 25,
            filename_suffix = f'modality_{modality_names[i]}'
        )

def calc_reconstruction_and_translation_mse(
    x: list[torch.Tensor],
    model,
    save_plot: bool = False,
    save_dir: pathlib.Path = pathlib.Path('.'),
    text_size: int = 30,
    vmin: float = 0,
    vmax: float = 0.1
):
    if save_plot:
        if not save_dir.exists():
            raise FileNotFoundError(f"Checkpoint dir {save_dir} doesn't exist.")

    num_modalities = len(x)

    translations = {}
    for input_modality in range(num_modalities):
        translations[input_modality] = {}
        for output_modality in range(num_modalities):
            translations[input_modality][output_modality] = model.translate(
                                                                x[input_modality],
                                                                input_modality=input_modality,
                                                                output_modality=output_modality
            )

    mse_matrix = np.zeros((num_modalities, num_modalities))
    for ii in range(num_modalities):
        for it in range(num_modalities):
            mse_matrix[ii, it] = np.mean((translations[ii][it] - x[it].cpu().numpy())**2)

    plt.figure(figsize=(8, 7))
    ax = sns.heatmap(mse_matrix, annot=True, fmt=".4f", cmap="coolwarm", linewidths=0.5, cbar=True, annot_kws={"size": text_size}, vmin=vmin, vmax=vmax)

    plt.xlabel("output modality", fontsize=text_size)
    plt.ylabel("input modality", fontsize=text_size)

    ticks = [str(i+1) for i in range(num_modalities)]
    ax.set_xticklabels(ticks, fontsize=text_size)
    ax.set_yticklabels(ticks, fontsize=text_size)

    # plt.title("MSE Heatmap", fontsize=text_size)

    # Customize colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=text_size)

    if save_plot:
        figure_file = save_dir / "reconstruction_translation_mse_matrix.svg"
        plt.savefig(figure_file, format='svg', bbox_inches='tight')
        plt.close()
    else:
        plt.show()