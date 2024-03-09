"""
Utilities for giving Matplotlib figures an Epoch style.

You might expect the coordinate transformation functions in here to be wrong.

You'd use this module like this:

    ```
    import matplotlib.pyplot as plt
    import egraphs

    # Set the theme once at the beginning of your script
    egraphs.set_epoch_theme()

    # Create your figure like you normally would
    plt.plot([1, 2, 3], [4, 5, 6])

    # Update the layout of the figure after creating it
    egraphs.relayout()

    plt.show()
    ```

or using the context manager like this

    ```
    import matplotlib.pyplot as plt
    import egraphs

    with egraphs.epoch_theme():
        # Create your figure like you normally would
        plt.plot([1, 2, 3], [4, 5, 6])

        # Update the layout of the figure after creating it
        egraphs.relayout()

        plt.show()
    ```
"""

from contextlib import contextmanager
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as patches
from matplotlib.collections import PolyCollection
import matplotlib as mpl
from matplotlib.path import Path
from matplotlib.backends.backend_agg import FigureCanvasAgg
from math import inf


frame_color        = '#CCD8D9'
tick_label_color   = '#5C737B'
text_color         = '#2B424B'
axis_label_color   = text_color
legend_label_color = text_color

epoch_gradient = [
  '#D7FDDD', '#C9F6D8', '#BAF0D3', '#ACE9CE', '#9EE3C9', '#8FDCC4',
  '#80D5BF', '#72CFBA', '#60C8B7', '#4DC1B3', '#3ABAB0', '#26B3AD',
  '#11ABA9', '#00A2AD', '#0099B1', '#0090B2', '#0088B3', '#007FB5',
  '#0776B4', '#156DB1', '#2364AD', '#2E5AA8', '#3152A1', '#35499A',
  '#384193', '#3A388B', '#393082', '#38287A', '#372071', '#351867',
  '#350D5F', '#340057',
]

# Property reference: https://matplotlib.org/stable/users/explain/customizing.html#customizing-with-style-sheets

epoch_rc = {
    'text': {
        'color': text_color,
    },

    'patch': {
        'linewidth': 0.5,
    },

    'xtick': {
        'labelsize': 11,
        'direction': 'out',
        'color': frame_color,
        'labelcolor': tick_label_color,
        'major.size': 6,
        'minor.size': 6,
    },

    'ytick': {
        'labelsize': 11,
        'direction': 'in',
        'color': frame_color,
        'labelcolor': tick_label_color,
        'major.size': 6,
        'minor.size': 6,
    },

    'lines': {
        'linewidth': 1.5,
    },

    'font': {
        'family': 'Messina Sans',
    },

    'grid': {
        'color': '#EBF5F4',
        'linestyle': '-',
        'linewidth': 1,
    },

    'axes': {
        'linewidth': 1,
        'edgecolor': frame_color,
        'labelcolor': axis_label_color,
        'labelsize': 12,
        'labelweight': 'bold',
        'grid': True,
        'axisbelow': True,
        'facecolor': 'white',
    },

    'legend': {
        'loc': 'upper right',
        'frameon': False,
        'handlelength': 1.4,
        'handleheight': 0.8,
        'borderaxespad': 0,
        'borderpad': 0,
        'fontsize': 10,
    },

    'figure': {
        'subplot.wspace': 0.05,
    }
}

processed_rc_params = {}
for key, value in epoch_rc.items():
    if isinstance(value, dict):
        for subkey, subvalue in value.items():
            processed_rc_params[f"{key}.{subkey}"] = subvalue
    else:
        processed_rc_params[key] = value

epoch_rc = processed_rc_params



def get_gradient_colors(n):
    """Returns a list of `n` colors from the Epoch gradient."""
    return [epoch_gradient[int(i/(n - 1) * (len(epoch_gradient) - 1))] for i in range(n)]

def px_to_pt(px):
    return px * 72 / 96

def in_to_px(inches, ppi=None):
    if ppi is None: ppi = mpl.rcParams['figure.dpi']
    return [inch * ppi for inch in inches] if isinstance(inches, tuple) else inches * ppi


def px_to_in(pixels, ppi=None):
    if ppi is None: ppi = mpl.rcParams['figure.dpi']
    return [pixel / ppi for pixel in pixels] if isinstance(pixels, tuple) else pixels / ppi


def px_to_x_fraction(px, ax=None):
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width * fig.dpi
    x_fraction = px / width

    return x_fraction


def px_to_y_fraction(px, ax=None):
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    height = bbox.height * fig.dpi
    y_fraction = px / height

    return y_fraction


@contextmanager
def epoch_theme():
    """Use this context manager to set the Epoch theme only for a specific block of code."""
    with mpl.rc_context(mpl.rcParamsDefault):
        with mpl.rc_context(epoch_rc):
            yield


def set_epoch_theme(dpi=None):
    """Sets the theme for Matplotlib to the Epoch style. Call this once at the beginning of your script."""
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams.update(epoch_rc)


def relayout(fig=None, replace_legend=False, legend={}, padding={}, xaxis={}, yaxis={}):
    """
    Updates the layout of the figure to match the Epoch style. Call this after creating your figure.

    If `replace_legend` is True, this function destroys the legend and manually recreates it (experimental).
    This is to give us more control over it in the future.
    """

    if fig is None:
        fig = plt.gcf()

    for ax in fig.axes:
        # Color the y-tick labels
        for label in ax.get_yticklabels(): label.set_weight('medium')
        for label in ax.get_xticklabels(): label.set_weight('medium')

        pixel_to_x_fraction = px_to_x_fraction(1, ax)
        pixel_to_y_fraction = px_to_y_fraction(1, ax)

        # Move the y axis label to the top-left corner
        if ax.get_ylabel():
            ax.yaxis.label.set_rotation(0)
            ax.yaxis.label.set_horizontalalignment('left')
            ax.yaxis.label.set_position((0.0, yaxis.get('labeloffset', 1.0 + 15 * pixel_to_y_fraction)))

        #ax.xaxis.set_label_coords(0.5, px_to_y_fraction(-40, ax))
        # set label pad
        ax.xaxis.labelpad = xaxis.get('labelpad', 10)

        # If there's a legend, make it horizontal and place it on top
        if ax.get_legend():
            if not replace_legend:
                # move the legend to the top
                ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.06), ncol=len(ax.get_legend().legendHandles), **legend)
            else:
                handles, labels = ax.get_legend_handles_labels()

                ax.get_legend().remove()

                symbol_width = legend.get('symbol_width', 12) * pixel_to_x_fraction
                item_spacing = legend.get('symbol_width', 8) * pixel_to_x_fraction
                symbol_label_spacing = 4 * pixel_to_x_fraction

                pos_x = 0
                pos_y = 1.065

                legend_items = []

                for item_index, (handle, label) in enumerate(zip(handles, labels)):
                    if isinstance(handle, mlines.Line2D):
                        color = handle.get_color()
                        dash = handle.get_linestyle()
                        offset, dash_pattern = handle._unscaled_dash_pattern
                        linestyle = (offset, [p/3 for p in dash_pattern]) if dash_pattern else '-'
                        symbol = mlines.Line2D([pos_x, pos_x + symbol_width], [pos_y, pos_y], color=color, linestyle=linestyle, linewidth=1.5, transform=ax.transAxes)
                    elif isinstance(handle, PolyCollection):
                        facecolor = handle.get_facecolor()[0]
                        edgecolor = handle.get_edgecolor()[0]
                        symbol_height = symbol_width / pixel_to_x_fraction * pixel_to_y_fraction
                        symbol = patches.Rectangle((pos_x, pos_y - 0.5 * symbol_height), symbol_width,
                                symbol_height, facecolor=facecolor, edgecolor=edgecolor, transform=ax.transAxes)
                    elif isinstance(handle, patches.Rectangle):
                        facecolor = handle.get_facecolor()
                        #edgecolor = handle.get_edgecolor()
                        symbol_height = symbol_width / pixel_to_x_fraction * pixel_to_y_fraction
                        symbol = patches.Rectangle((pos_x, pos_y - 0.5 * symbol_height), symbol_width,
                                symbol_height, facecolor=facecolor, transform=ax.transAxes)
                    else:
                        raise NotImplementedError(f'Handle for {type(handle)} not implemented')

                    ax.add_artist(symbol)
                    symbol.set_clip_on(False)
                    legend_items.append(symbol)
                    ax.add_line(mlines.Line2D([0, 1], [pos_y, pos_y], color='black', linestyle='--', transform=ax.transAxes))

                    pos_x += symbol_width
                    pos_x += symbol_label_spacing
                    
                    v_center_adjustment = -0.8 * pixel_to_y_fraction # Matplotlib's centering is not great
                    text = ax.text(pos_x, pos_y + v_center_adjustment, label, size=10, transform=ax.transAxes, verticalalignment='center', color=legend_label_color, fontweight='medium')
                    text_width = ax.transAxes.inverted().transform_bbox(text.get_window_extent()).width
                    legend_items.append(text)

                    pos_x += text_width

                    if item_index < len(handles) - 1:
                        pos_x += item_spacing

                max_x = max([ax.transAxes.inverted().transform_bbox(item.get_window_extent()).x1 for item in legend_items])

                for item in legend_items:
                    # Displace them to the right border
                    # TODO: This is probably a very crappy way to do this

                    displacement = 1 - max_x

                    if isinstance(item, mlines.Line2D):
                        x_data = item.get_xdata()
                        item.set_xdata([x_data[0] + displacement, x_data[1] + displacement])
                    elif isinstance(item, patches.Rectangle):
                        x_data = item.get_x()
                        item.set_x(x_data + displacement)
                    elif isinstance(item, mpl.text.Text):
                        item.set_x(item.get_position()[0] + displacement)

    # Tight layout while maintaining the figure dimensions
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.canvas.draw()

    min_x0 = inf
    max_x1 = -inf
    min_y0 = inf
    max_y1 = -inf

    for ax in fig.axes:
        bounds = ax.get_tightbbox(fig.canvas.get_renderer())
        min_x0 = min(min_x0, bounds.x0)
        max_x1 = max(max_x1, bounds.x1)
        min_y0 = min(min_y0, bounds.y0)
        max_y1 = max(max_y1, bounds.y1)

        if ax.get_xlabel():
            bounds = ax.xaxis.label.get_tightbbox(fig.canvas.get_renderer())
            min_x0 = min(min_x0, bounds.x0)
            max_x1 = max(max_x1, bounds.x1)
            min_y0 = min(min_y0, bounds.y0)
            max_y1 = max(max_y1, bounds.y1)

        if ax.get_ylabel():
            bounds = ax.yaxis.label.get_tightbbox(fig.canvas.get_renderer())
            min_x0 = min(min_x0, bounds.x0)
            max_x1 = max(max_x1, bounds.x1)
            min_y0 = min(min_y0, bounds.y0)
            max_y1 = max(max_y1, bounds.y1)

    size_px = in_to_px(fig.get_size_inches(), ppi=fig.dpi)

    # paddings in display units
    left_padding   = padding.get('left', 20)
    right_padding  = padding.get('right', 20)
    top_padding    = padding.get('top', 20)
    bottom_padding = padding.get('bottom', 20)

    min_x0 -= left_padding
    max_x1 += right_padding
    min_y0 -= bottom_padding
    max_y1 += top_padding

    bbox_w = max_x1 - min_x0
    bbox_h = max_y1 - min_y0

    bottom = -min_y0 / bbox_h
    top = (size_px[1] - min_y0) / bbox_h

    left = -min_x0 / bbox_w
    right = (size_px[0] - min_x0) / bbox_w

    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)


def add_brace(ax, left, bottom, top, transform=None, linewidth=1, color='black'):
    """
    Add a brace. Call this after relayout().

    It's experimental right now!
    """

    font = {}
    bool_auto = True

    if transform:
        # force an update
        ax.get_xbound()
        ax.get_ybound()

        left, top = transform.transform((left, top))
        right, bottom = transform.transform((0, bottom))

    # Extracted from an SVG
    points = [
        [0.     , 0.     ], # MOVETO [0]
        [0.     , 0.     ], # LINETO [1]

        [0.37308, 0.     ], # CURVE4 [2]
        [0.67555, 0.30246], # CURVE4 [3]
        [0.67555, 0.67556], # CURVE4 [4]

        [0.67555, 2.99492], # LINETO [5]

        [0.67555, 3.23316], # CURVE4 [6]
        [0.77621, 3.46029], # CURVE4 [7]
        [0.95272, 3.62035], # CURVE4 [8]

        [0.95272, 3.62035], # LINETO [9]

        # middle point

        [0.95272, 3.62035], # LINETO [10]

        [0.77621, 3.7804 ], # CURVE4 [11]
        [0.67555, 4.00753], # CURVE4 [12]
        [0.67555, 4.24578], # CURVE4 [13]

        [0.67555, 6.56515], # LINETO [14]

        [0.67555, 6.93823], # CURVE4 [15]
        [0.37308, 7.2407 ], # CURVE4 [16]
        [0.     , 7.2407 ], # CURVE4 [17]

        [0.     , 7.2407 ], # LINETO [18]
    ]

    target_width = 15

    for point in points:
        point[0] *= target_width
        point[1] *= target_width

    # Increase the height of the brace by augmenting the length of the straight segments
    target_height = top - bottom

    current_height = points[-1][1]
    remaining_height = target_height - current_height

    # increase first line segment length
    for i in range(5, len(points)):
        points[i][1] += remaining_height/2

    # increase second line segment length
    for i in range(14, len(points)):
        points[i][1] += remaining_height/2

    for point in points:
        point[0] += left
        point[1] += bottom

    commands = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,

        # middle point

        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
    ]

    # Convert to axes coordinates
    for point in points:
        #point[0], point[1] = ax.transData.transform((point[0], point[1]))
        # apply inverse
        point[0], point[1] = ax.transAxes.inverted().transform((point[0], point[1]))

    path = Path(points, commands)

    ax.add_patch(patches.PathPatch(path, facecolor='none', linewidth=linewidth, edgecolor=color, zorder=10, transform=ax.transAxes, clip_on=False))
