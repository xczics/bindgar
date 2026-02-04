from typing import List, Tuple, Optional
from matplotlib.axes import Axes
import numpy as np  # type: ignore
from numpy import ndarray
from .physics import M_SUN, M_EARTH

def scatter_positions(ax: Axes, xym_arrays: ndarray, **kwargs) -> None:
    """Scatter plot of x-y positions sizes by mass.

    Args:
        ax (Axes): Matplotlib Axes object to plot on.
        xym_arrays (ndarray): Numpy array of shape (N, 3) where each row is (x, y, mass).
    """
    x = xym_arrays[:, 0]
    y = xym_arrays[:, 1]
    masses = xym_arrays[:, 2]
    ref_mass = kwargs.pop('ref_mass', min(masses))
    set_equal = kwargs.pop('set_equal', True)
    size_scale = kwargs.pop('size_scale', 2)

    sizes = (masses / ref_mass) ** (2/3) * size_scale  # Scale sizes by mass^(2/3)

    max_r = max((x**2 + y**2)**0.5)
    print(f"Max radius in scatter_positions: {max_r}")
    if set_equal:
        ax.set_aspect('equal')
    ax.set_xlim(-max_r*1.1, max_r*1.1)
    ax.set_ylim(-max_r*1.1, max_r*1.1)
    split_colors_by_mass(masses, kwargs)
    ax.scatter(x, y, s=sizes, **kwargs)

    ax.set_xlabel('x (a.u.)')
    ax.set_ylabel('y (a.u.)')

def split_colors_by_mass(m_array: ndarray,
                         kwargs: dict) -> None:
    """Helper function to split colors by mass threshold.
    """
    if 'm_split' in kwargs and m_array is not None and ('c' not in kwargs and 'color' not in kwargs):
        m_split = kwargs.pop('m_split')
        if 'color_small' in kwargs and 'color_large' in kwargs:
            color_small = kwargs.pop('color_small')
            color_large = kwargs.pop('color_large')
            colors = [color_small if m <= m_split else color_large for m in m_array] 
        else:
            colors = ['blue' if m <= m_split else 'orange' for m in m_array]
        kwargs['c'] = colors
    

def scatter_ae(ax: Axes, 
               a_array: ndarray, e_array: ndarray, 
               m_array: Optional[ndarray] = None,
               **kwargs) -> None:
    """Scatter plot of semi-major axis vs eccentricity.
    Args:
        ax (Axes): Matplotlib Axes object to plot on.
        a_array (ndarray): Numpy array of semi-major axes.
        e_array (ndarray): Numpy array of eccentricities.
        m_array (ndarray, optional): Numpy array of masses for size scaling. Defaults to None.
    """
    if m_array is not None:
        ref_mass = kwargs.pop('ref_mass', min(m_array))
        size_scale = kwargs.pop('size_scale', 2)
        sizes = (m_array / ref_mass) ** (2/3) * size_scale
    else:
        sizes = np.full_like(a_array, 8)
    if m_array is not None:
        split_colors_by_mass(m_array, kwargs)
    ax.scatter(a_array, e_array, s=sizes, **kwargs)
    ax.set_xlabel('Semi-major axis (a.u.)')
    ax.set_ylabel('Eccentricity')
    ax.set_xlim(0, max(a_array)*1.1)
    ax.set_ylim(0, max(e_array)*1.1)

def scatter_am(ax: Axes, 
               a_array: ndarray, m_array: ndarray, 
               **kwargs) -> None:
    """Scatter plot of semi-major axis vs mass.
    Args:
        ax (Axes): Matplotlib Axes object to plot on.
        a_array (ndarray): Numpy array of semi-major axes.
        m_array (ndarray): Numpy array of masses.
    """
    m_array = m_array.copy()
    ref_mass = kwargs.pop('ref_mass', min(m_array))
    size_scale = kwargs.pop('size_scale', 2)
    sizes = (m_array / ref_mass) ** (2/3) * size_scale
    max_mas = kwargs.pop('max_mas', 200.0)
    split_colors_by_mass(m_array, kwargs)
    m_array /= M_EARTH / M_SUN  # Convert mass to M_earth
    if max_mas == -1:
        max_mas = max(m_array)
    ax.scatter(a_array, m_array, s=sizes, **kwargs)
    ax.set_xlabel('Semi-major axis (a.u.)')
    ax.set_ylabel(r'Mass ($M_{earth}$)')
    ax.set_xlim(0, max(a_array)*1.1)
    ax.set_ylim(min(m_array)*0.85, min(max_mas, max(m_array)*1.1))
    ax.set_yscale('log')

