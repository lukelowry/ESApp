"""
Shared plotting helpers for ESAplus example notebooks.

These functions encapsulate common visualization patterns so that
notebook cells remain focused on core esapp operations. Import from
a hidden cell near the top of each notebook::

    import sys; sys.path.insert(0, '..')
    from plot_helpers import plot_barh_top, plot_direction_sensitivity, ...
"""

import numpy as np
import matplotlib.pyplot as plt

# Attempt to import esapp plotting utilities (available when esapp is installed)
try:
    from esapp.utils import format_plot, border, plot_lines, plot_vecfield, darker_hsv_colormap
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Generic chart helpers
# ---------------------------------------------------------------------------

def plot_barh_top(values, labels=None, n=20, title='', xlabel='', ylabel='',
                  color='steelblue', figsize=(10, 5), ax=None):
    """Horizontal bar chart of the top-*n* items sorted descending."""
    top = values[:n] if len(values) <= n else values.sort_values(ascending=False).head(n)
    if labels is None:
        labels = [f'{i+1}' for i in range(len(top))]
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(top)), top.values if hasattr(top, 'values') else top,
            color=color)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels[:len(top)])
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


def plot_dual_bar(values_a, values_b, label_a='A', label_b='B',
                  xlabel='Index', ylabel='Value', title='', figsize=(10, 5)):
    """Grouped bar chart comparing two datasets side-by-side."""
    fig, ax = plt.subplots(figsize=figsize)
    x = range(len(values_a))
    width = 0.35
    ax.bar([i - width / 2 for i in x], values_a, width,
           label=label_a, color='steelblue', alpha=0.8)
    ax.bar([i + width / 2 for i in x], values_b, width,
           label=label_b, color='tomato', alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_hist(values, bins=20, title='', xlabel='', ylabel='Count',
              color='steelblue', ax=None):
    """Simple histogram with white edge on bars."""
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(values, bins=bins, color=color, edgecolor='white')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


# ---------------------------------------------------------------------------
# Power system specific
# ---------------------------------------------------------------------------

def plot_voltage_profile(vmag, vang=None, figsize=(14, 5)):
    """Scatter of bus voltage magnitudes (with optional angle stem plot)."""
    ncols = 2 if vang is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=figsize)
    if ncols == 1:
        axes = [axes]

    axes[0].scatter(range(len(vmag)), vmag, c='steelblue', s=30, edgecolors='white')
    axes[0].axhline(y=0.95, color='red', linestyle='--', alpha=0.7, label='0.95 pu limit')
    axes[0].axhline(y=1.05, color='red', linestyle='--', alpha=0.7, label='1.05 pu limit')
    axes[0].axhline(y=1.0, color='grey', linestyle='-', alpha=0.3)
    axes[0].set_xlabel('Bus Index')
    axes[0].set_ylabel('Voltage Magnitude (pu)')
    axes[0].set_title('Bus Voltage Profile')
    axes[0].legend()

    if vang is not None:
        axes[1].stem(range(len(vang)), vang, linefmt='steelblue', markerfmt='o', basefmt=' ')
        axes[1].set_xlabel('Bus Index')
        axes[1].set_ylabel('Voltage Angle (degrees)')
        axes[1].set_title('Bus Voltage Angles')

    plt.tight_layout()
    plt.show()


def plot_branch_loading(branches_loaded, figsize=(10, 5)):
    """Horizontal bar chart of most-loaded branches."""
    fig, ax = plt.subplots(figsize=figsize)
    labels = [f"{int(r['BusNum'])}-{int(r['BusNum:1'])}" for _, r in branches_loaded.iterrows()]
    ax.barh(range(len(branches_loaded)), branches_loaded['LinePercent'].values,
            color='steelblue')
    ax.set_yticks(range(len(branches_loaded)))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.axvline(x=100, color='red', linestyle='--', alpha=0.7, label='100% limit')
    ax.set_xlabel('Loading (%)')
    ax.set_title('Top 15 Most Loaded Branches')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_gen_dispatch_and_voltage(online_gens, bus_data, figsize=(14, 5)):
    """Generator MW bar chart + bus voltage scatter."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    gen_mw = online_gens.sort_values('GenMW', ascending=True)
    axes[0].barh(range(len(gen_mw)), gen_mw['GenMW'].values, color='steelblue')
    axes[0].set_yticks(range(len(gen_mw)))
    axes[0].set_yticklabels([f"Bus {b}" for b in gen_mw['BusNum']])
    axes[0].set_xlabel('MW Output')
    axes[0].set_title('Generator Active Power Dispatch')

    axes[1].scatter(bus_data['BusNum'], bus_data['BusPUVolt'],
                    c='steelblue', s=40, edgecolors='white')
    axes[1].axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='0.95 pu')
    axes[1].axhline(y=1.05, color='red', linestyle='--', alpha=0.5, label='1.05 pu')
    axes[1].set_xlabel('Bus Number')
    axes[1].set_ylabel('Voltage (pu)')
    axes[1].set_title('Bus Voltage Profile')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_gen_load_balance(total_gen, total_load, figsize=(6, 4)):
    """Bar chart comparing total generation vs. total load."""
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(['Generation', 'Load'], [total_gen, total_load],
                  color=['steelblue', 'tomato'])
    ax.set_ylabel('MW')
    ax.set_title('System Generation vs Load Balance')
    for bar, val in zip(bars, [total_gen, total_load]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


def plot_contingency_results(violations, figsize=(14, 5)):
    """Bar chart of violations per contingency + histogram."""
    if len(violations) == 0 or 'Contingency' not in violations.columns:
        return
    ctg_counts = violations['Contingency'].value_counts().head(15)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].barh(range(len(ctg_counts)), ctg_counts.values, color='steelblue')
    axes[0].set_yticks(range(len(ctg_counts)))
    axes[0].set_yticklabels(ctg_counts.index, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_xlabel('Number of Violations')
    axes[0].set_title('Top 15 Contingencies by Violation Count')

    axes[1].hist(violations.groupby('Contingency').size(), bins=20,
                 color='steelblue', edgecolor='white')
    axes[1].set_xlabel('Violations per Contingency')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Distribution of Violations')

    plt.tight_layout()
    plt.show()


def plot_pv_curve(mw_points, v_points, figsize=(10, 6)):
    """PV curve with nose point marker."""
    if not mw_points:
        return
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(mw_points, v_points, 'o-', color='steelblue', markersize=3)

    nose_idx = np.argmax(mw_points)
    ax.plot(mw_points[nose_idx], v_points[nose_idx], 'r*', markersize=15,
            label=f'Nose point: {mw_points[nose_idx]:.0f} MW')

    ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.5, label='0.95 pu limit')
    format_plot(ax, title='PV Curve (Voltage Stability)',
                xlabel='Interface Transfer (MW)',
                ylabel='Critical Bus Voltage (pu)',
                plotarea='white')
    ax.legend()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Sparse matrix / spectral
# ---------------------------------------------------------------------------

def plot_spy_matrices(matrices, titles, figsize=None, markersize=3, colors=None):
    """Side-by-side spy() plots for one or more sparse matrices."""
    n = len(matrices)
    if figsize is None:
        figsize = (6 * n, 5)
    if colors is None:
        colors = ['steelblue'] * n
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, M, t, c in zip(axes, matrices, titles, colors):
        ax.spy(M, markersize=markersize, color=c)
        ax.set_title(t)
    plt.tight_layout()
    plt.show()


def plot_ybus_analysis(Y, figsize=(12, 5)):
    """Y-Bus sparsity pattern + eigenvalue spectrum."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].spy(Y, markersize=3, color='steelblue')
    axes[0].set_title(f'Y-Bus Sparsity Pattern\n{Y.shape}, nnz={Y.nnz}')

    eig_Y = np.linalg.eigvals(Y.toarray())
    axes[1].scatter(eig_Y.real, eig_Y.imag, s=20, c='steelblue', edgecolors='white')
    axes[1].set_xlabel('Real')
    axes[1].set_ylabel('Imaginary')
    axes[1].set_title('Y-Bus Eigenvalue Spectrum')
    axes[1].axhline(y=0, color='grey', linewidth=0.5)
    axes[1].axvline(x=0, color='grey', linewidth=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_incidence_and_degree(A, figsize=(12, 5)):
    """Incidence matrix spy + bus degree bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].spy(A, markersize=2, color='steelblue')
    axes[0].set_title(f'Incidence Matrix\n{A.shape}')
    axes[0].set_xlabel('Bus index')
    axes[0].set_ylabel('Branch index')

    degrees = np.abs(A).T @ np.ones(A.shape[0])
    axes[1].bar(range(len(degrees)), degrees, color='steelblue')
    format_plot(axes[1], title='Bus Degree Distribution',
                xlabel='Bus index', ylabel='Degree', plotarea='white')

    plt.tight_layout()
    plt.show()


def plot_incidence_and_laplacian(A, figsize=(12, 5)):
    """Incidence matrix spy + |A.T @ A| image."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].spy(A, markersize=2, color='steelblue')
    axes[0].set_title(f'Incidence Matrix\n{A.shape}')
    axes[0].set_xlabel('Bus index')
    axes[0].set_ylabel('Branch index')

    L_unw = (A.T @ A).toarray()
    axes[1].imshow(np.abs(L_unw), cmap='Blues', aspect='auto')
    axes[1].set_title('|A.T @ A| (Unweighted Laplacian)')
    axes[1].set_xlabel('Bus index')
    axes[1].set_ylabel('Bus index')

    plt.tight_layout()
    plt.show()


def plot_eigenspectrum(eigenvalue_sets, titles, figsize=None):
    """Stem plots of one or more eigenvalue arrays."""
    n = len(eigenvalue_sets)
    if figsize is None:
        figsize = (6 * n, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, vals, t in zip(axes, eigenvalue_sets, titles):
        ax.stem(vals, basefmt=' ')
        format_plot(ax, title=t, xlabel='Index', ylabel='Eigenvalue',
                    plotarea='white')
    plt.tight_layout()
    plt.show()


def plot_fiedler(fiedler, figsize=(8, 5)):
    """Fiedler vector bar chart colored by sign."""
    fig, ax = plt.subplots(figsize=figsize)
    colors = ['steelblue' if v >= 0 else 'tomato' for v in fiedler]
    ax.bar(range(len(fiedler)), fiedler, color=colors)
    format_plot(ax, title='Fiedler Vector (Natural Network Partition)',
                xlabel='Bus index', ylabel='Fiedler component', plotarea='white')
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_histograms(datasets, titles, xlabels, colors=None, bins=25, figsize=None):
    """Side-by-side histograms."""
    n = len(datasets)
    if figsize is None:
        figsize = (6 * n, 4)
    if colors is None:
        colors = ['steelblue', 'tomato', 'seagreen', 'goldenrod'][:n]
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, data, t, xl, c in zip(axes, datasets, titles, xlabels, colors):
        ax.hist(data, bins=bins, color=c, edgecolor='white')
        format_plot(ax, title=t, xlabel=xl, ylabel='Count', plotarea='white')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Direction sensitivity (GIC)
# ---------------------------------------------------------------------------

def plot_direction_sensitivity(directions, max_gics, title='Max GIC vs. Storm Direction',
                               figsize=(14, 5)):
    """Line plot + polar plot of GIC vs. storm direction."""
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(121)
    ax1.plot(directions, max_gics, 'o-', color='steelblue', markersize=3)
    ax1.set_xlabel('Storm Direction (degrees from North)')
    ax1.set_ylabel('Max |GIC| (Amps)')
    ax1.set_title(title)
    ax1.grid(True)

    ax2 = fig.add_subplot(122, projection='polar')
    theta = np.radians(directions)
    ax2.plot(theta, max_gics, 'o-', color='tomato', markersize=3)
    ax2.set_title('GIC Polar Response', pad=20)

    plt.tight_layout()
    plt.show()

    worst = directions[np.argmax(max_gics)]
    print(f"Worst-case direction: {worst} degrees")
    print(f"Worst-case max GIC: {max_gics.max():.2f} Amps")


def plot_direction_profiles(directions, gic_profiles, labels, figsize=(14, 5)):
    """Multi-transformer direction sensitivity: line + polar."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for j, lbl in enumerate(labels):
        axes[0].plot(directions, gic_profiles[:, j], label=lbl)
    format_plot(axes[0], title='Top Transformer GIC vs. Storm Direction',
                xlabel='Direction (deg from N)', ylabel='|GIC| (Amps)',
                plotarea='white')
    axes[0].legend()

    ax_polar = fig.add_axes(axes[1].get_position(), projection='polar')
    axes[1].set_visible(False)
    theta = np.radians(directions)
    for j, lbl in enumerate(labels):
        ax_polar.plot(theta, gic_profiles[:, j], label=lbl)
    ax_polar.set_title('Polar GIC Response', pad=20)
    ax_polar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=8)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# GIC matrix / sensitivity
# ---------------------------------------------------------------------------

def plot_gic_distribution(gic_abs, n=20, figsize=(14, 5)):
    """Bar chart of top-*n* transformer GICs + histogram."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    top = gic_abs.sort_values(ascending=False).head(n)
    axes[0].barh(range(len(top)), top.values, color='steelblue')
    axes[0].set_yticks(range(len(top)))
    axes[0].set_yticklabels([f'XF {i + 1}' for i in range(len(top))])
    axes[0].invert_yaxis()
    axes[0].set_xlabel('|GIC| (Amps)')
    axes[0].set_title(f'Top {n} Transformer GICs')

    axes[1].hist(gic_abs, bins=20, color='steelblue', edgecolor='white')
    axes[1].set_xlabel('|GIC| (Amps)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('GIC Magnitude Distribution')

    plt.tight_layout()
    plt.show()


def plot_gic_bar_hist(gic_abs, n=15, figsize=(14, 5)):
    """Histogram + top-N bar chart for GIC magnitudes."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(gic_abs, bins=20, color='steelblue', edgecolor='white')
    format_plot(axes[0], title='Distribution of Transformer GIC Magnitudes',
                xlabel='|GIC| (Amps)', ylabel='Count', plotarea='white')

    top = gic_abs.sort_values(ascending=False).head(n)
    axes[1].barh(range(len(top)), top.values, color='steelblue')
    axes[1].set_yticks(range(len(top)))
    axes[1].set_yticklabels([f'XF {i + 1}' for i in range(len(top))])
    axes[1].invert_yaxis()
    format_plot(axes[1], title=f'Top {n} Transformer GICs',
                xlabel='|GIC| (Amps)', ylabel='Transformer', plotarea='white')

    plt.tight_layout()
    plt.show()


def plot_gmatrix_comparison(G_model, G_pw, figsize=(16, 4)):
    """Compare model G-matrix vs PowerWorld G-matrix with difference."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    im0 = axes[0].imshow(np.abs(G_model), cmap='Blues', aspect='auto')
    fig.colorbar(im0, ax=axes[0], shrink=0.7)
    axes[0].set_title('|G| from model()')

    im1 = axes[1].imshow(np.abs(G_pw), cmap='Blues', aspect='auto')
    fig.colorbar(im1, ax=axes[1], shrink=0.7)
    axes[1].set_title('|G| from PowerWorld')

    if G_model.shape == G_pw.shape:
        diff = np.abs(G_model - G_pw)
        im2 = axes[2].imshow(diff, cmap='Reds', aspect='auto')
        fig.colorbar(im2, ax=axes[2], shrink=0.7)
        axes[2].set_title(f'|Difference|\nmax={diff.max():.2e}')
    else:
        axes[2].text(0.5, 0.5, 'Different shapes\n(different node sets)',
                     ha='center', va='center', transform=axes[2].transAxes)
        axes[2].set_title('Difference')

    plt.tight_layout()
    plt.show()


def plot_jacobian_sensitivity(J_dense, figsize=(14, 5)):
    """dI/dE Jacobian heatmap + row-wise sensitivity bar chart."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    im0 = axes[0].imshow(np.abs(J_dense), cmap='Blues', aspect='auto')
    fig.colorbar(im0, ax=axes[0], shrink=0.7)
    axes[0].set_xlabel('Branch index')
    axes[0].set_ylabel('Transformer index')
    axes[0].set_title('|dI/dE| Jacobian')

    row_sens = np.sum(np.abs(J_dense), axis=1)
    axes[1].barh(range(len(row_sens)), row_sens, color='steelblue')
    axes[1].set_ylabel('Transformer index')
    axes[1].set_xlabel('Total sensitivity')
    axes[1].set_title('Transformer Sensitivity to E-Field')
    axes[1].invert_yaxis()

    plt.tight_layout()
    plt.show()


def plot_branch_impact(col_sens, top_n=5, figsize=(12, 4)):
    """Bar chart of branch voltage impact with top branches highlighted."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(col_sens)), col_sens, color='steelblue', width=1.0)
    format_plot(ax, title='Branch Voltage Impact on Total GIC',
                xlabel='Branch index', ylabel='Aggregate |dI/dE|',
                plotarea='white')

    top_branches = np.argsort(col_sens)[::-1][:top_n]
    for b in top_branches:
        ax.bar(b, col_sens[b], color='tomato', width=1.0)

    plt.tight_layout()
    plt.show()

    print(f"\nTop {top_n} most influential branches: {top_branches}")


# ---------------------------------------------------------------------------
# Geographic / E-field
# ---------------------------------------------------------------------------

def plot_geo_grid_buses(LON, LAT, lon, lat, shape, xlim, ylim, figsize=(10, 8)):
    """Grid points + bus locations on a geographic border."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(LON.ravel(), LAT.ravel(), s=1, c='lightblue', alpha=0.5,
               label='Grid points')
    ax.scatter(lon, lat, s=20, c='red', zorder=5, label='Bus locations')
    border(ax, shape)
    ax.set_xlim(xlim[0] - 0.1, xlim[1] + 0.1)
    ax.set_ylim(ylim[0] - 0.1, ylim[1] + 0.1)
    format_plot(ax, title='Geographic Grid with Bus Locations',
                xlabel=r'Longitude ($^\circ$E)', ylabel=r'Latitude ($^\circ$N)',
                plotarea='white', grid=False)
    ax.legend()
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_efield_comparison(LON, LAT, fields, shape, figsize=(18, 5)):
    """Side-by-side magnitude heatmaps for multiple E-field patterns.

    Parameters
    ----------
    fields : list of (name, Ex, Ey) tuples
    """
    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, (name, Ex, Ey) in zip(axes, fields):
        magnitude = np.sqrt(Ex ** 2 + Ey ** 2)
        im = ax.pcolormesh(LON, LAT, magnitude, cmap='hot_r', shading='auto')
        border(ax, shape)
        xlim = (LON.min(), LON.max())
        ylim = (LAT.min(), LAT.max())
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        fig.colorbar(im, ax=ax, label='|E| (V/km)', shrink=0.8)
        format_plot(ax, title=f'{name} |E|',
                    xlabel=r'Longitude ($^\circ$E)',
                    ylabel=r'Latitude ($^\circ$N)',
                    plotarea='white', grid=False)
        ax.set_aspect('equal')
    plt.suptitle('Electric Field Magnitude Comparison', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_efield_vectors(LON, LAT, Ex, Ey, shape, step=3, figsize=(10, 8)):
    """Heatmap of E-field magnitude + vector field overlay."""
    fig, ax = plt.subplots(figsize=figsize)
    magnitude = np.sqrt(Ex ** 2 + Ey ** 2)
    im = ax.pcolormesh(LON, LAT, magnitude, cmap='YlOrRd', shading='auto', alpha=0.6)
    border(ax, shape)
    ax.set_xlim(LON.min(), LON.max())
    ax.set_ylim(LAT.min(), LAT.max())

    sm = plot_vecfield(ax, LON[::step, ::step], LAT[::step, ::step],
                       Ex[::step, ::step], Ey[::step, ::step],
                       scale=40, width=0.003)
    fig.colorbar(im, ax=ax, label='|E| (V/km)', shrink=0.8)
    format_plot(ax, title='Spatially Varying E-Field',
                xlabel=r'Longitude ($^\circ$E)',
                ylabel=r'Latitude ($^\circ$N)', grid=False)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_network_efield(LON, LAT, magnitude, lines, lon, lat, Ex, Ey,
                        shape, step=4, figsize=(12, 9)):
    """Full network overlay: heatmap + transmission lines + buses + E-field vectors."""
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.pcolormesh(LON, LAT, magnitude, cmap='YlOrRd', shading='auto', alpha=0.4)
    border(ax, shape)
    plot_lines(ax, lines, ms=8, lw=0.8)
    ax.scatter(lon, lat, s=30, c='navy', zorder=6, label='Buses',
               edgecolors='white', linewidth=0.5)
    ax.quiver(LON[::step, ::step], LAT[::step, ::step],
              Ex[::step, ::step], Ey[::step, ::step],
              color='darkred', alpha=0.7, scale=30, width=0.002, zorder=7)
    ax.set_xlim(LON.min(), LON.max())
    ax.set_ylim(LAT.min(), LAT.max())

    fig.colorbar(im, ax=ax, label='|E| (V/km)', shrink=0.7)
    format_plot(ax, title='E-Field with Transmission Network Overlay',
                xlabel=r'Longitude ($^\circ$E)',
                ylabel=r'Latitude ($^\circ$N)',
                plotarea='white', grid=False)
    ax.legend(loc='lower right')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_gic_geo_map(lines, xf_geo, gic_mag, shape, xlim, ylim, figsize=(12, 9)):
    """GIC magnitudes on a geographic map with transmission network."""
    fig, ax = plt.subplots(figsize=figsize)
    border(ax, shape)
    plot_lines(ax, lines, ms=5, lw=0.5)

    sizes = 20 + 200 * gic_mag / gic_mag.max()
    sc = ax.scatter(xf_geo['Longitude'], xf_geo['Latitude'],
                    s=sizes, c=gic_mag, cmap='Reds', zorder=8,
                    edgecolors='black', linewidth=0.5)
    fig.colorbar(sc, ax=ax, label='|GIC| (Amps)', shrink=0.7)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    format_plot(ax, title='Transformer GIC Magnitudes on Geographic Map',
                xlabel=r'Longitude ($^\circ$E)',
                ylabel=r'Latitude ($^\circ$N)',
                plotarea='white', grid=False)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_b3d_roundtrip(LON, LAT, ex_orig, ex_loaded, shape, ny, nx,
                       figsize=(14, 5)):
    """Side-by-side original vs loaded Ex from B3D."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ex_2d_orig = ex_orig[0].reshape(ny, nx, order='F')
    ex_2d_load = ex_loaded[0].reshape(ny, nx, order='F')

    for ax, data, title in zip(axes,
                               [ex_2d_orig, ex_2d_load],
                               ['Original Ex', 'B3D Round-Trip Ex']):
        im = ax.pcolormesh(LON, LAT, data, cmap='RdBu_r', shading='auto')
        border(ax, shape)
        ax.set_xlim(LON.min(), LON.max())
        ax.set_ylim(LAT.min(), LAT.max())
        fig.colorbar(im, ax=ax, label='Ex (V/km)')
        format_plot(ax, title=title,
                    xlabel=r'Longitude ($^\circ$E)',
                    ylabel=r'Latitude ($^\circ$N)',
                    plotarea='white', grid=False)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_b3d_components(LON, LAT, Ex, Ey, shape, suptitle='', figsize=(18, 5)):
    """Three-panel plot: |E|, Ex, Ey on a geographic background."""
    magnitude = np.sqrt(Ex ** 2 + Ey ** 2)
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    for ax, data, cmap, label, title in zip(
        axes,
        [magnitude, Ex, Ey],
        ['hot_r', 'RdBu_r', 'RdBu_r'],
        ['|E| (V/km)', 'Ex (V/km)', 'Ey (V/km)'],
        ['E-Field Magnitude', 'Ex (Eastward)', 'Ey (Northward)'],
    ):
        im = ax.pcolormesh(LON, LAT, data, cmap=cmap, shading='auto')
        border(ax, shape)
        fig.colorbar(im, ax=ax, label=label, shrink=0.8)
        format_plot(ax, title=title,
                    xlabel=r'Longitude ($^\circ$E)',
                    ylabel=r'Latitude ($^\circ$N)',
                    plotarea='white', grid=False)
        ax.set_aspect('equal')

    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Dynamics
# ---------------------------------------------------------------------------

def plot_comparative_dynamics(ctg_names, all_results, figsize=None):
    """Stacked subplots of generator power for each contingency."""
    if figsize is None:
        figsize = (12, 4 * len(ctg_names))
    fig, axes = plt.subplots(len(ctg_names), 1, figsize=figsize, sharex=True)
    if len(ctg_names) == 1:
        axes = [axes]
    for ax, name in zip(axes, ctg_names):
        results = all_results[name]
        p_cols = [c for c in results.columns if 'P' in str(c) or 'MW' in str(c)]
        if p_cols:
            results[p_cols].plot(ax=ax, legend=True)
        format_plot(ax, title=f'{name}: Generator Power',
                    xlabel='Time (s)', ylabel='P (MW)', plotarea='white')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Discrete calculus / Grid2D utilities
# ---------------------------------------------------------------------------

def plot_grid_regions(X, Y, grid, figsize=(14, 4)):
    """Three-panel view of grid: all points, boundary/interior, edges.

    Parameters
    ----------
    grid : Grid2D
        Grid2D instance (provides .boundary, .interior, .left, etc.).
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    xf = X.ravel(order='C')
    yf = Y.ravel(order='C')

    axes[0].scatter(xf, yf, s=5, c='steelblue')
    axes[0].set_title('All Grid Points')
    axes[0].set_aspect('equal')

    axes[1].scatter(xf[grid.interior], yf[grid.interior],
                    s=5, c='steelblue', label='Interior')
    axes[1].scatter(xf[grid.boundary], yf[grid.boundary],
                    s=10, c='tomato', label='Boundary')
    axes[1].set_title('Boundary vs Interior')
    axes[1].set_aspect('equal')
    axes[1].legend(markerscale=2)

    axes[2].scatter(xf[grid.left], yf[grid.left],
                    s=10, c='red', label='Left')
    axes[2].scatter(xf[grid.right], yf[grid.right],
                    s=10, c='blue', label='Right')
    axes[2].scatter(xf[grid.top], yf[grid.top],
                    s=10, c='green', label='Top')
    axes[2].scatter(xf[grid.bottom], yf[grid.bottom],
                    s=10, c='orange', label='Bottom')
    axes[2].set_title('Edge Selectors')
    axes[2].set_aspect('equal')
    axes[2].legend(markerscale=2)

    plt.tight_layout()
    plt.show()


def plot_scalar_field(X, Y, f, title='', clabel='f(x,y)', cmap='RdBu_r',
                      figsize=(6, 5)):
    """Single pcolormesh of a scalar field with colorbar."""
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.pcolormesh(X, Y, f, cmap=cmap, shading='auto')
    fig.colorbar(im, ax=ax, label=clabel)
    format_plot(ax, title=title, xlabel='x', ylabel='y', grid=False,
                plotarea='white')
    plt.tight_layout()
    plt.show()


def plot_field_panels(X, Y, fields, titles, cmap='RdBu_r', figsize=None,
                      suptitle=None, equal_aspect=True):
    """Row of pcolormesh panels, one per field.

    Parameters
    ----------
    fields : list of 2-D arrays
    titles : list of str
    """
    n = len(fields)
    if figsize is None:
        figsize = (5 * n + 1, 4)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, data, t in zip(axes, fields, titles):
        im = ax.pcolormesh(X, Y, data, cmap=cmap, shading='auto')
        fig.colorbar(im, ax=ax)
        ax.set_title(t)
        if equal_aspect:
            ax.set_aspect('equal')
    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_gradient_vecfield(X, Y, f, grad_x, grad_y, step=3, figsize=(6, 5)):
    """Scalar field background + gradient vector field overlay."""
    Xs = X[::step, ::step]
    Ys = Y[::step, ::step]
    Us = grad_x[::step, ::step]
    Vs = grad_y[::step, ::step]

    fig, ax = plt.subplots(figsize=figsize)
    ax.pcolormesh(X, Y, f, cmap='Greys', shading='auto', alpha=0.3)
    sm = plot_vecfield(ax, Xs, Ys, Us, Vs, scale=150, width=0.003)
    fig.colorbar(sm, ax=ax, label='Angle (rad)')
    format_plot(ax, title='Gradient Vector Field', xlabel='x', ylabel='y',
                grid=False)
    plt.tight_layout()
    plt.show()


def plot_div_curl(X, Y, u_field, v_field, div_uv, curl_uv, step=3,
                  figsize=(16, 4)):
    """Quiver plot of vector field + divergence + curl pcolormesh."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].quiver(X[::step, ::step], Y[::step, ::step],
                   u_field[::step, ::step], v_field[::step, ::step],
                   color='steelblue')
    axes[0].set_title('Vector Field (u, v)')
    axes[0].set_aspect('equal')

    im1 = axes[1].pcolormesh(X, Y, div_uv, cmap='RdBu_r', shading='auto')
    fig.colorbar(im1, ax=axes[1])
    axes[1].set_title('Divergence (expansion = 1.0)')
    axes[1].set_aspect('equal')

    im2 = axes[2].pcolormesh(X, Y, curl_uv, cmap='RdBu_r', shading='auto')
    fig.colorbar(im2, ax=axes[2])
    axes[2].set_title('Curl (rotation = 1.0)')
    axes[2].set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_hodge_rotation(X, Y, f, grad_x, grad_y, rot_x, rot_y, step=3,
                        figsize=(11, 5)):
    """Side-by-side quiver plots: gradient vs Hodge-rotated gradient."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].pcolormesh(X, Y, f, cmap='Greys', shading='auto', alpha=0.3)
    axes[0].quiver(X[::step, ::step], Y[::step, ::step],
                   grad_x[::step, ::step], grad_y[::step, ::step],
                   color='steelblue')
    axes[0].set_title('Gradient Field')
    axes[0].set_aspect('equal')

    axes[1].pcolormesh(X, Y, f, cmap='Greys', shading='auto', alpha=0.3)
    axes[1].quiver(X[::step, ::step], Y[::step, ::step],
                   rot_x[::step, ::step], rot_y[::step, ::step],
                   color='tomato')
    axes[1].set_title('Hodge Star (90-degree Rotation)')
    axes[1].set_aspect('equal')

    plt.tight_layout()
    plt.show()


def plot_eigenmodes(X, Y, vals, vecs, ny, nx, k=9, figsize=(12, 10)):
    """3x3 grid of Laplacian eigenmodes."""
    fig, axes = plt.subplots(3, 3, figsize=figsize)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(vals):
            ax.set_visible(False)
            continue
        mode = vecs[:, i].reshape(ny, nx)
        ax.pcolormesh(X, Y, mode, cmap='RdBu_r', shading='auto')
        ax.set_title(f'Mode {i}, eig={vals[i]:.3f}')
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.suptitle('Laplacian Eigenmodes', fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Spectral analysis utilities
# ---------------------------------------------------------------------------

def plot_vecfield_gallery(X, Y, fields, step=3, figsize=(11, 10)):
    """2x2 vector field gallery using plot_vecfield.

    Parameters
    ----------
    fields : dict of {name: (U, V)} tuples
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    for ax, (name, (U, V)) in zip(axes.ravel(), fields.items()):
        sm = plot_vecfield(ax, X[::step, ::step], Y[::step, ::step],
                           U[::step, ::step], V[::step, ::step],
                           scale=30, width=0.004)
        format_plot(ax, title=name, xlabel='x', ylabel='y', grid=False)
    plt.tight_layout()
    plt.show()


def plot_graph_operators(matrices, titles, cmaps=None, vranges=None,
                         suptitle='', figsize=None):
    """Grid of imshow plots for graph matrices."""
    n = len(matrices)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols
    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)
    if cmaps is None:
        cmaps = ['RdBu_r'] * n
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.array(axes).ravel() if n > 1 else [axes]
    for i, (ax, M, t, cm) in enumerate(zip(axes_flat, matrices, titles, cmaps)):
        kwargs = {'cmap': cm, 'aspect': 'auto'}
        if vranges and i < len(vranges) and vranges[i]:
            kwargs['vmin'], kwargs['vmax'] = vranges[i]
        ax.imshow(M, **kwargs)
        ax.set_title(t)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
    # Hide extra axes
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)
    if suptitle:
        plt.suptitle(suptitle, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_normlap_spectrum(L_norm, evals, figsize=(11, 4)):
    """Normalized Laplacian image + eigenvalue stem plot."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].imshow(L_norm, cmap='RdBu_r')
    axes[0].set_title('Normalized Cycle Laplacian')

    axes[1].stem(evals, basefmt=' ')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Eigenvalue')
    axes[1].set_title('Eigenvalue Spectrum')
    axes[1].axhline(y=2, color='r', linestyle='--', alpha=0.5, label='eig=2 bound')
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def plot_hermitify(M, H, figsize=(10, 4)):
    """Side-by-side |M| vs |H| images."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(np.abs(M), cmap='viridis')
    axes[0].set_title('|M| (complex symmetric)')
    axes[1].imshow(np.abs(H), cmap='viridis')
    axes[1].set_title('|H| (Hermitian)')
    plt.tight_layout()
    plt.show()


def plot_colormap_scales(scales, figsize=(14, 3)):
    """Show darker_hsv_colormap at different scales."""
    fig, axes = plt.subplots(1, len(scales), figsize=figsize)
    if len(scales) == 1:
        axes = [axes]
    gradient = np.linspace(-np.pi, np.pi, 256).reshape(1, -1)
    for ax, scale in zip(axes, scales):
        cmap = darker_hsv_colormap(scale)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_title(f'darker_hsv_colormap(scale={scale})')
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()


def plot_colormap_2d(LON, LAT, theta, scales, figsize=(15, 6)):
    """1D gradient + 2D angle field at multiple colormap scales."""
    gradient = np.linspace(-np.pi, np.pi, 256).reshape(1, -1)

    fig, axes = plt.subplots(2, len(scales), figsize=figsize)
    for ax, scale in zip(axes[0], scales):
        cmap = darker_hsv_colormap(scale)
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        ax.set_title(f'scale={scale}')
        ax.set_yticks([])
    for ax, scale in zip(axes[1], scales):
        cmap = darker_hsv_colormap(scale)
        ax.pcolormesh(LON, LAT, theta, cmap=cmap, shading='auto')
        ax.set_title(f'Angle field (scale={scale})')
        ax.set_aspect('equal')
    plt.suptitle('darker_hsv_colormap at Different Scales', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_borders(shapes, figsize=(16, 5)):
    """Side-by-side geographic borders."""
    fig, axes = plt.subplots(1, len(shapes), figsize=figsize)
    if len(shapes) == 1:
        axes = [axes]
    for ax, shape in zip(axes, shapes):
        border(ax, shape)
        format_plot(ax, title=f'{shape} Border',
                    xlabel=r'Longitude ($^\circ$E)',
                    ylabel=r'Latitude ($^\circ$N)',
                    plotarea='white', grid=False)
        ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_network_map(lines, lon, lat, shape, pad=0.5, figsize=(10, 8)):
    """Transmission network on geographic background."""
    fig, ax = plt.subplots(figsize=figsize)
    border(ax, shape)
    plot_lines(ax, lines, ms=15, lw=1.2)
    ax.set_xlim(lon.min() - pad, lon.max() + pad)
    ax.set_ylim(lat.min() - pad, lat.max() + pad)
    format_plot(ax, title='Transmission Network',
                xlabel=r'Longitude ($^\circ$E)',
                ylabel=r'Latitude ($^\circ$N)',
                plotarea='white', grid=False)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_bus_voltages_map(lines, lon, lat, vmag, shape, pad=0.5,
                          figsize=(10, 8)):
    """Bus voltages colored on geographic map with network overlay."""
    fig, ax = plt.subplots(figsize=figsize)
    border(ax, shape)
    plot_lines(ax, lines, ms=5, lw=0.8)

    sc = ax.scatter(lon, lat, s=50, c=vmag, cmap='RdYlGn', vmin=0.95, vmax=1.05,
                    zorder=6, edgecolors='black', linewidth=0.5)
    fig.colorbar(sc, ax=ax, label='Voltage (pu)', shrink=0.7)

    ax.set_xlim(lon.min() - pad, lon.max() + pad)
    ax.set_ylim(lat.min() - pad, lat.max() + pad)
    format_plot(ax, title='Bus Voltages on Geographic Map',
                xlabel=r'Longitude ($^\circ$E)',
                ylabel=r'Latitude ($^\circ$N)',
                plotarea='white', grid=False)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_vecfield_map(LON, LAT, Ex, Ey, lines, shape, figsize=(10, 8)):
    """Vector field over network with geographic border."""
    fig, ax = plt.subplots(figsize=figsize)
    border(ax, shape)
    plot_lines(ax, lines, ms=3, lw=0.5)

    sm = plot_vecfield(ax, LON, LAT, Ex, Ey, scale=30, width=0.003)
    fig.colorbar(sm, ax=ax, label='Angle (rad)', shrink=0.7)

    ax.set_xlim(LON.min(), LON.max())
    ax.set_ylim(LAT.min(), LAT.max())
    format_plot(ax, title='Synthetic Vector Field over Network',
                xlabel=r'Longitude ($^\circ$E)',
                ylabel=r'Latitude ($^\circ$N)',
                grid=False)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


def plot_format_showcase(x_data, figsize=(16, 4)):
    """Showcase of format_plot styling options."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(x_data, np.sin(x_data), 'o-', markersize=3)
    format_plot(axes[0], title='Default Style', xlabel='x', ylabel='sin(x)')

    axes[1].plot(x_data, np.cos(x_data), 'o-', markersize=3, color='steelblue')
    format_plot(axes[1], title='Colored Background', xlabel='x', ylabel='cos(x)',
                plotarea='#f0f0f0')

    axes[2].plot(x_data, np.sin(x_data) * np.exp(-x_data / 5), 'o-', markersize=3)
    format_plot(axes[2], title='Custom Ticks', xlabel='x', ylabel='y',
                xlim=(0, 10), ylim=(-1, 1), xticksep=2.5, yticksep=0.5)

    plt.tight_layout()
    plt.show()
