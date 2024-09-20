from typing import List, Tuple, Union
import seaborn as sns
import matplotlib as mpl

Palette = Union[str, List[Union[str, Tuple[float, float, float]]], None]


def get_colormap(palette: Palette) -> mpl.colors.Colormap:
    """Generate a matplotlib Colormap from a `Palette`"""

    if palette is None:
        return sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)
    elif isinstance(palette, str):
        return mpl.colormaps[palette]
    elif isinstance(palette, list):
        return mpl.colors.ListedColormap(palette)
