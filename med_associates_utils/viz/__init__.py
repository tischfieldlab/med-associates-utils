import matplotlib as mpl

from .cumulative import plot_cumulative_events
from .raster import plot_event_raster

# set some matplotlib parameters
mpl.rcParams['pdf.fonttype'] = 42  # turn on true-type fonts, provides better editing of PDF/Illustrator
