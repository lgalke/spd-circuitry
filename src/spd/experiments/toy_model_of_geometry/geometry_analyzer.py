

from pathlib import Path
import einops
from einops import reduce
import matplotlib.pyplot as plt
from  matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from simplex_dataset import SimplexDataset


def get_group_features(
    ranks: list[int],
) -> tuple[list[list[int]], list[int], int]:
    group_sizes = [k * (k + 1) for k in ranks]
    n_features = sum(group_sizes)

    groups = []
    start = 0
    for size in group_sizes:
        groups.append(list(range(start, start + size)))
        start += size

    return groups, group_sizes, n_features


def save_figure(fig: plt.Figure, save_path: Path, dpi: int = 350) -> None:
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_input_hidden_output_translation(
    W1,
    W2,
    group_features,
    save_dir,
    title=None,
    threshold=0.05,
):
    """
    Visualize how input groups activate hidden neurons,
    and how those neurons route to outputs.
    """

    n_groups = len(group_features)
    n_hidden = W1.shape[0]
    n_outputs = W2.shape[0]

    A = np.zeros((n_groups, n_hidden))
    B = W2


    for g, feature_ids in enumerate(group_features):
        #(hidden, features)
        group_input = W1[:, feature_ids]

        #sum over the feature group, to find the total activation of each hidden neuron from this group.
        hidden_activation = group_input.sum(axis=1)
        #for visual clarity cap at zero, albeit a negative result is not possible due to the ReLU in the original model.
        A[g, :] = np.maximum(hidden_activation, 0)

    #Across every group, pick the one that maximally activates it.
    dominant_input = np.argmax(A, axis=0)
    dominant_output = np.argmax(B, axis=0)
    input_strength = A.max(axis=0)
    output_strength = B.max(axis=0)
    keep = (input_strength >= threshold) & (output_strength >= threshold)

    order = np.lexsort((dominant_output, dominant_input))
    neuron_order = order[keep[order]]
    A_sorted = A[:, neuron_order]
    B_sorted = B[:, neuron_order]

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1,
        figsize=(16, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [n_groups, n_outputs]},
    )

    #As
    im_top = ax_top.imshow(A_sorted, aspect="auto", cmap="Blues")
    ax_top.set_title("Input group contribution to hidden neurons", fontsize=14)
    ax_top.set_yticks(range(n_groups))
    ax_top.set_yticklabels([f"group {g+1}" for g in range(n_groups)], fontsize=12)

    cax_top = make_axes_locatable(ax_top).append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im_top, cax=cax_top)

    # B
    im_bottom = ax_bottom.imshow(B_sorted, aspect="auto", cmap="Reds")
    ax_bottom.set_xlabel("Hidden neurons", fontsize=12)
    ax_bottom.set_yticks(range(n_outputs))
    ax_bottom.set_yticklabels([f"output {i+1}" for i in range(n_outputs)], fontsize=12)
    ax_bottom.set_title("Hidden neuron contribution to outputs", fontsize=14)

    cax_bottom = make_axes_locatable(ax_bottom).append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im_bottom, cax=cax_bottom)

    sorted_dominant_input = dominant_input[neuron_order]
    group_sizes = np.bincount(sorted_dominant_input, minlength=n_groups)
    boundaries = np.cumsum(group_sizes)[:-1]
    for boundary in boundaries:
        ax_top.axvline(boundary - 0.5, color="black", lw=1)
        ax_bottom.axvline(boundary - 0.5, color="black", lw=1)

    if title:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    save_figure(fig, save_dir / "w1_w2_input_hidden_output_translation.png")
    plt.close(fig)

def plot_group_output_matrix(
    W_in,                 # W_in : inner matrix, shape (hidden, n_features)
    W_out,                 # W_out: outer matrix, shape (n_out, hidden)
    dataset,            # SimplexDataset
    save_dir,
    title=None,
    n_samples=200,
):

    device = dataset.device
    W_in  = torch.as_tensor(W_in, dtype=torch.float32, device=device)
    W_out = torch.as_tensor(W_out, dtype=torch.float32, device=device)

    n_groups = dataset.output_dim
    n_out    = W_out.shape[0]
    P = np.zeros((n_groups, n_out))

    with torch.no_grad():
        for g, (k, group) in enumerate(zip(dataset.dimensions, dataset.groups)):
            vertices = dataset._sample_valid_simplices(n_samples, k)

            X = torch.zeros(n_samples, dataset.n_features, device=device)
            X[:, group] = vertices.reshape(n_samples, -1)   # Couldve also just changed the data distribution to only one active, but its okay.

            # forward pass
            H = torch.relu(X @ W_in.T)
            Y = H @ W_out.T

            P[g] = Y.mean(dim=0).cpu().numpy()

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    im = ax.imshow(P, cmap="viridis", aspect="auto")

    if title:
        ax.set_title(title)
    ax.set_xlabel("Output group")
    ax.set_ylabel("Input group")
    ax.set_xticks(range(n_out))
    ax.set_yticks(range(n_groups))
    ax.set_xticklabels([f"out{i+1}" for i in range(n_out)])
    ax.set_yticklabels([f"g{i+1}" for i in range(n_groups)])

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    save_figure(fig, save_dir / title)
    plt.close(fig)

def compute_io_flows(C1, C2, group_features):
    """
    Computes the read, write and subcomponent matrix as defined in the thesis 
    C1 has shape (C, d_hid, d_in) and C2 has shape (C, d_out, d_hid)
    """

    read_per_group = []

    #C1 = (C, d_hid, d_in)
    #C2 = (C, d_out, d_hid)
    for g in group_features:
        C1_g = C1[:, :, g]          # (C, d_hid, |g|)
        C1_g_sum = np.sum(C1_g, axis=-1)     # (C, d_hid)
        read_per_group.append(C1_g_sum)

    read = np.stack(read_per_group, axis=-1)  # (C, d_hid, n_groups)
    read = np.sum(read, axis=-2)              # (C, n_groups)

    write = C2.sum(axis=-1)  # (C2, n_outputs)

    in_h = np.sum(C1, axis=-1)   # (C, d_hid)
    out_h = np.sum(C2, axis=-2)  # (C, d_hid)
    overlap = in_h @ out_h.T            # (C, C)
    
    return read, overlap, write


def pick_components(read, write, coverage=0.95):
    """
    Picks components that together contribute a fraction of the total layer magnitude.
    """
    contrib1 = np.abs(read).sum(axis=1)    # (C1,) input-side inflow
    contrib2 = np.abs(write).sum(axis=1)   # (C2,) output-side outflow

    def cover(v):
        order = np.argsort(-v)
        cum = np.cumsum(v[order])
        total = cum[-1]

        if total <= 0:
            raise ValueError("Total mass is zero or negative.")
        
        n = np.searchsorted(cum, coverage * total) + 1
        return order[:n]

    return cover(contrib1), cover(contrib2)


def build_columns(read, write, n_groups, n_outputs, c1, c2):
    return [
        {
            "title": "Input groups",
            "ids": [i+1 for i in range(n_groups)],
            "imp": read.sum(0),
            "order": list(range(n_groups)),
        },
        {
            "title": "W1 subcomponents",
            "ids": [str(i) for i in c1],
            "imp": read.sum(1),
            "order": list(range(len(c1))),    
        },
        {
            "title": "W2 subcomponents",
            "ids": [str(i) for i in c2],
            "imp": write.sum(1),
            "order": list(range(len(c2))),
        },
        {
            "title": "Output groups",
            "ids": [i+1 for i in range(n_outputs)],
            "imp": write.sum(0),
            "order": list(range(n_outputs)),
        },
    ]


def barycenter_sweep(columns, edges, sweeps=8):
    """
    Some sorting algorithm from stack exchange. Not too important,
    as it only serves to disentangle the visualization.
    """
    sizes = [len(c["ids"]) for c in columns]

    for c in columns:
        c["order"] = list(range(len(c["ids"])))

    for _ in range(sweeps):
        for col, nbr, mat in edges:

            nrank = np.empty(sizes[nbr])
            nrank[columns[nbr]["order"]] = np.arange(sizes[nbr])

            w = mat.sum(1)

            bc = (mat @ nrank) / np.where(w > 0, w, 1)

            cur = np.empty(sizes[col])
            cur[columns[col]["order"]] = np.arange(sizes[col])

            bc = np.where(w > 0, bc, cur)

            columns[col]["order"] = list(np.argsort(bc, kind="stable"))

    return columns

def layout_column(imp, order, gap=0.03):

    n = len(imp)

    #Every "node" is the same size.
    total_mass = 1 - gap * (n - 1)
    each = total_mass / n
    h = np.full(n, each)


    y = 1.0
    centers = np.empty(n)
    for i in order:
        centers[i] = y - (h[i] / 2)
        y -= h[i] + gap

    return centers, h


def render_io_chain(columns, flows, save_path, title=None, edge_frac=0.05):
    xs = [0, 1, 2, 3]
    bw = 0.045

    flow_colors = ["#378ADD", "#7F77DD", "#D85A30"]

    fig, ax = plt.subplots(figsize=(13, 7))

    def ribbon(x0, y0, x1, y1, lw, color, alpha):
        xm = (x0 + x1) / 2
        path = MplPath(
            [(x0, y0), (xm, y0), (xm, y1), (x1, y1)],
            [MplPath.MOVETO, MplPath.CURVE4,
             MplPath.CURVE4, MplPath.CURVE4],
        )
        ax.add_patch(PathPatch(
            path,
            fill=False,
            lw=lw,
            edgecolor=color,
            alpha=alpha,
            capstyle="round",
        ))

    # flows
    for i, M in enumerate(flows):
        #Scale proportional to the max in a layer, im not sure if this is the best way of doing it...
        max_flow = np.abs(M).max()
        if max_flow <= 0:
            raise ValueError("Flow matrix has non-positive maximum value, cannot scale ribbons.")
        #horizontal starting, end points bw= beam width
        x0 = xs[i] + bw / 2
        x1 = xs[i + 1] - bw / 2

        threshold = edge_frac * max_flow
        active_sources, active_targets = np.where(np.abs(M) >= threshold)

        for s, d in zip(active_sources, active_targets):

            flow_value = M[s, d]
            r = flow_value / max_flow #this normalizes to 0 <= 1 <= 1
            r= np.abs(r)

            ribbon(
                x0,
                columns[i]["centers"][s],
                x1,
                columns[i + 1]["centers"][d],
                0.5 + 7 * r,
                flow_colors[i],
                0.15 + 0.5 * np.abs(r),
            )

    for i, col in enumerate(columns):
        for j, label in enumerate(col["ids"]):
            yc = col["centers"][j]
            h = col["heights"][j]

            ax.add_patch(Rectangle(
                (xs[i] - bw / 2, yc - h / 2),
                bw, h,
                facecolor="#B5D4F4" if i < 2 else "#F5C4B3",
                edgecolor="#185FA5" if i < 2 else "#993C1D",
                lw=0.8,
                zorder=3,
            ))

            ax.text(xs[i], yc, label, ha="center", va="center", fontsize=12)

        ax.text(xs[i], 1.06, col["title"], ha="center", fontsize=14)

    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-0.1, 1.13)
    ax.axis("off")

    if title:
        fig.suptitle(title, y=0.9, fontsize=16)

    save_figure(fig, save_path)
    plt.close(fig)


def plot_io_routing_chain(
    C1, C2, group_features, save_dir,
    title=None, coverage=0.5,
    edge_frac=0.01, sort_nodes=True, sweeps=20,
):
    read, overlap, write = compute_io_flows(C1, C2, group_features)
    #Read has dim (C,_n groups), overlap ahs dim (C1, C2), write has dim (C2, n_outputs)

    #Select only the components that matter significantly
    c1, c2 = pick_components(
        read, write,
        coverage=coverage,
    )

    #pick components that satisfy minimum mass
    read = read[c1, :]        
    write = write[c2, :]   
    
    #Likewise for the overlap, only the components that carry mass.
    overlap = overlap[np.ix_(c1, c2)]

    columns = build_columns(
        read, write,
        len(group_features),
        C2.shape[1],
        c1, c2,
    )

    if sort_nodes:
        edges = [
        (1, 0, read),
        (2, 1, overlap.T),
        (2, 3, write),
        (1, 2, overlap),
    ]
        columns = barycenter_sweep(columns, edges, sweeps)
  
    for col in columns:
        col["centers"], col["heights"] = layout_column(col["imp"], col["order"])

    render_io_chain(
        columns,
        [read.T, overlap, write],
        save_dir / "io_routing_chain.png",
        title,
        edge_frac,
    )


def main() -> None:
    #I have not had the time to clean this up fully, so i apologize in advance.
      
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ranks = [2,2,2]
    group_features, _, _ = get_group_features(ranks)
    dataset = SimplexDataset(ranks, device=device)

    run_dirs = [
        (r"C:\Users\Knud\uni\spd\spd\experiments\toy_model_of_geometry\out\minimality_sweep\0.1", "RSPD -Minimality coefficient: 1e-1"),
    ]

    #Subcomponent level
    for run_dir, run_title in run_dirs:
        RUN_DIR = Path(run_dir)
        state_dict = torch.load(
            RUN_DIR / "model_40000.pth",
            map_location=device,
        )

            
        A1 = state_dict["components.linear1.A"]
        B1 = state_dict["components.linear1.B"]
        A2 = state_dict["components.linear2.A"]
        B2 = state_dict["components.linear2.B"]

        #Components
        C1 = einops.einsum(A1, B1, "d_in C K, C K d_out -> C d_out d_in").detach().cpu().numpy()
        C2 = einops.einsum(A2, B2, "d_in C K, C K d_out -> C d_out d_in").detach().cpu().numpy()
        plot_io_routing_chain(C1, C2, group_features, save_dir=RUN_DIR, title=run_title, coverage=0.80, edge_frac=0.01,sort_nodes=True, sweeps=10)


    #Target Model level
    model_dir = r"C:\Users\Knud\uni\spd_original\spd\experiments\toy_model_of_geometry\out\smaller_test"
    model_dir = Path(model_dir)
    state_dict = torch.load(
        model_dir / "geometry.pth",
        map_location=device,
    )
    W1 = state_dict["linear1.weight"].detach().cpu().numpy()
    W2 = state_dict["linear2.weight"].detach().cpu().numpy()

    plot_input_hidden_output_translation(W1, W2, group_features, save_dir=model_dir,
                                            threshold=0.05)
    plot_group_output_matrix(W1, W2, save_dir=model_dir, title="Input to output matrix (Target Model)", dataset=dataset)

if __name__ == "__main__":
    main()  