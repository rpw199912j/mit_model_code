import copy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pymatgen.core.periodic_table import Element


def periodic_table_heatmap(
    elemental_data,
    cbar_label="",
    cbar_label_size=14,
    show_plot=False,
    cmap="YlOrRd",
    cmap_range=None,
    blank_color="grey",
    value_format=None,
    max_row=9,
):
    """
    Modified from https://github.com/materialsproject/pymatgen/blob/v2022.0.6/pymatgen/util/plotting.py#L175
    A static method that generates a heat map overlayed on a periodic table.
    Args:
         elemental_data (dict): A dictionary with the element as a key and a
            value assigned to it, e.g. surface energy and frequency, etc.
            Elements missing in the elemental_data will be grey by default
            in the final table elemental_data={"Fe": 4.2, "O": 5.0}.
         cbar_label (string): Label of the colorbar. Default is "".
         cbar_label_size (float): Font size for the colorbar label. Default is 14.
         cmap_range (tuple): Minimum and maximum value of the colormap scale.
            If None, the colormap will autotmatically scale to the range of the
            data.
         show_plot (bool): Whether to show the heatmap. Default is False.
         value_format (str): Formatting string to show values. If None, no value
            is shown. Example: "%.4f" shows float to four decimals.
         cmap (string): Color scheme of the heatmap. Default is 'YlOrRd'.
            Refer to the matplotlib documentation for other options.
         blank_color (string): Color assigned for the missing elements in
            elemental_data. Default is "grey".
         max_row (integer): Maximum number of rows of the periodic table to be
            shown. Default is 9, which means the periodic table heat map covers
            the first 9 rows of elements.
    """
    # Create a copy of cmap to avoid the global state of cmap
    cmap = copy.copy(mpl.cm.get_cmap(cmap))

    # Convert primitive_elemental data in the form of numpy array for plotting.
    if cmap_range is not None:
        max_val = cmap_range[1]
        min_val = cmap_range[0]
    else:
        max_val = max(elemental_data.values())
        min_val = min(elemental_data.values())

    max_row = min(max_row, 9)

    if max_row <= 0:
        raise ValueError("The input argument 'max_row' must be positive!")

    value_table = np.empty((max_row, 18)) * np.nan
    blank_value = min_val - 0.01

    for el in Element:
        if el.row > max_row:
            continue
        value = elemental_data.get(el.symbol, blank_value)
        value_table[el.row - 1, el.group - 1] = value

    fig, ax = plt.subplots()
    plt.gcf().set_size_inches(12, 8)

    # We set nan type values to masked values (ie blank spaces)
    data_mask = np.ma.masked_invalid(value_table.tolist())
    heatmap = ax.pcolor(
        data_mask,
        cmap=cmap,
        edgecolors="w",
        linewidths=1,
        vmin=min_val - 0.001,
        vmax=max_val + 0.001,
    )
    cbar = fig.colorbar(heatmap)

    # Grey out missing elements in input data
    cbar.cmap.set_under(blank_color)

    # Set the colorbar label and tick marks
    cbar.set_label(cbar_label, rotation=90, labelpad=25, size=cbar_label_size)
    cbar.ax.tick_params(labelsize=cbar_label_size)

    # Refine and make the table look nice
    ax.axis("off")
    ax.invert_yaxis()

    # Label each block with corresponding element and value
    for i, row in enumerate(value_table):
        for j, el in enumerate(row):
            if not np.isnan(el):
                symbol = Element.from_row_and_group(i + 1, j + 1).symbol
                if symbol == "O":
                    plt.text(
                        j + 0.5,
                        i + 0.25,
                        symbol,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=14,
                        color="white"
                    )
                    if el != blank_value and value_format is not None:
                        plt.text(
                            j + 0.5,
                            i + 0.5,
                            value_format % el,
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=10,
                            color="white"
                        )
                else:
                    plt.text(
                        j + 0.5,
                        i + 0.25,
                        symbol,
                        horizontalalignment="center",
                        verticalalignment="center",
                        fontsize=14,
                    )
                    if el != blank_value and value_format is not None:
                        plt.text(
                            j + 0.5,
                            i + 0.5,
                            value_format % el,
                            horizontalalignment="center",
                            verticalalignment="center",
                            fontsize=10,
                        )

    plt.tight_layout()

    if show_plot:
        plt.show()

    return ax
