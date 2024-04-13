# -*- coding: utf-8 -*-

"""
Plot Formatters

This module implements helpful classes to format your plots or create meshes.

This code is pulled from the open-source package pyswarms by @ljvmiranda921 with
minimal adaptation for the purpose of visualizing hybrid optimization results.
"""

# Import modules
import numpy as np
from attr import attrib, attrs
from attr.validators import instance_of
from matplotlib import cm, colors
import pandas as pd
import sys
import multiprocessing

if sys.platform != 'win32':
    multiprocessing.set_start_method('fork')


@attrs
class Designer(object):
    """Designer class for specifying a plot's formatting and design

    You can use this class for specifying design-related customizations to
    your plot. This can be passed in various functions found in the
    :mod:`pyswarms.utils.plotters` module.

    .. code-block :: python

        from pyswarms.utils.plotters import plot_cost_history
        from pyswarms.utils.plotters.formatters import Designer

        # Set title_fontsize into 20
        my_designer = Designer(title_fontsize=20)

        # Assuming we already had an optimizer ready
        plot_cost_history(cost_history, designer=my_designer)

    Attributes
    ----------
    figsize : tuple
        Overall figure size. Default is `(10, 8)`
    title_fontsize : str, int, or float
        Size of the plot's title. Default is `large`
    text_fontsize : str, int, or float
        Size of the plot's labels and legend. Default is `medium`
    legend : str
        Label to show in the legend. For cost histories, it states
        the label of the line plot. Default is `Cost`
    label : array_like
        Label to show in the x, y, or z-axis. For a 3D plot, please pass an
        iterable with three elements. Default is :code:`['x-axis', 'y-axis',
        'z-axis']`
    limits : list
        The x-, y-, z- limits of the axes. Pass an iterable with the number of
        elements representing the number of axes. Default is :code:`[(-1, 1),
        (-1, 1), (-1, 1)]`
    colormap : matplotlib.cm.Colormap
        Colormap for contour plots. Default is `cm.viridis`
    """

    # Overall plot design
    figsize = attrib(type=tuple, validator=instance_of(tuple), default=(10, 8))
    title_fontsize = attrib(validator=instance_of((str, int, float)), default="large")
    text_fontsize = attrib(validator=instance_of((str, int, float)), default="medium")
    legend = attrib(validator=instance_of(str), default="Cost")
    label = attrib(
        validator=instance_of((str, list, tuple)),
        default=["x-axis", "y-axis", "z-axis"],
    )
    limits = attrib(
        validator=instance_of((list, tuple)),
        default=[(-1, 1), (-1, 1), (-1, 1)],
    )
    colormap = attrib(validator=instance_of(colors.Colormap), default=cm.viridis)


@attrs
class Animator(object):
    """Animator class for specifying animation behavior

    You can use this class to modify options on how the animation will be run
    in the :func:`pyswarms.utils.plotters.plot_contour` and
    :func:`pyswarms.utils.plotters.plot_surface` methods.

    .. code-block :: python

        from pyswarms.utils.plotters import plot_contour
        from pyswarms.utils.plotters.formatters import Animator

        # Do not repeat animation
        my_animator = Animator(repeat=False)

        # Assuming we already had an optimizer ready
        plot_contour(pos_history, animator=my_animator)

    Attributes
    ----------
    interval : int
        Sets the interval or speed into which the animation is played.
        Default is `80`
    repeat_delay : int or float, optional
        Sets the delay before repeating the animation again.
    repeat : bool, optional
        Pass `False` if you don't want to repeat the animation.
        Default is `True`
    """

    interval = attrib(type=int, validator=instance_of(int), default=80)
    repeat_delay = attrib(default=None)
    repeat = attrib(type=bool, validator=instance_of(bool), default=True)


@attrs
class Mesher(object):
    """Mesher class for plotting contours of objective functions

    This class enables drawing a surface plot of a given objective function.
    You can customize how this plot is drawn with this class. Pass an instance
    of this class to enable meshing.

    .. code-block :: python

        from pyswarms.utils.plotters import plot_surface
        from pyswarms.utils.plotters.formatters import Mesher
        from pyswarms.utils.functions import single_obj as fx

        # Use sphere function
        my_mesher = Mesher(func=fx.sphere)

        # Assuming we already had an optimizer ready
        plot_surface(pos_history, mesher=my_mesher)

    Attributes
    ----------
    func : callable
        Objective function to plot a surface of.
    delta : float
        Number of steps when generating the surface plot
        Default is `0.001`
    limits : list or tuple
        The range, in each axis, where the mesh will be drawn.
        Default is :code:`[(-1,1), (-1,1)]`
    levels : list or int, optional
        Levels on which the contours are shown. If :code:`int` is passed,
        then `matplotlib` automatically computes for the level positions.
        Default is :code:`numpy.arange(-2.0, 2.0, 0.070)`
    alpha : float, optional
        Transparency of the surface plot. Default is `0.3`
    limits : list, optional
        The x-, y-, z- limits of the axes. Pass an iterable with the number of
        elements representing the number of axes. Default is :code:`[(-1, 1),
        (-1, 1)]`
    """

    func = attrib()
    # For mesh creation
    delta = attrib(type=float, default=0.001)
    limits = attrib(validator=instance_of((list, tuple)), default=[(-1, 1), (-1, 1)])
    levels = attrib(type=list, default=np.arange(-2.0, 2.0, 0.070))
    # Surface transparency
    alpha = attrib(type=float, validator=instance_of(float), default=0.3)

    def compute_history_3d(self, pos_history, n_processes=None):
        """Compute a 3D position matrix

        The first two columns are the 2D position in the x and y axes
        respectively, while the third column is the fitness on that given
        position.

        Parameters
        ----------
        pos_history : numpy.ndarray
            Two-dimensional position matrix history of shape
            :code:`(iterations, n_particles, 2)`
        n_processes : int
        number of processes to use for parallel mesh point calculation (default: None = no parallelization)

        Returns
        -------
        numpy.ndarray
            3D position matrix of shape :code:`(iterations, n_particles, 3)`
        """

        # Setup Pool of processes for parallel evaluation
        pool = None if n_processes is None else mp.Pool(n_processes)

        if pool is None:
            fitness = np.array(list(map(self.func, pos_history)))
        else:
            iter_r = []
            # Iterate over iterations
            for i in range(len(pos_history)):

                # Parallelize particles
                r_map_split = pool.map(
                    self.func,
                    np.array_split(np.array(pos_history[i]), pool._processes),
                )
                iter_r.append(np.array(np.concatenate(r_map_split)))
            fitness = np.array(iter_r)

        # Close Pool of Processes
        if n_processes is not None:
            pool.close()

        return np.dstack((pos_history, fitness))


# -*- coding: utf-8 -*-

"""
Plotting tool for Optimizer Analysis -- pulled from PySwarms

This module is built on top of :code:`matplotlib` to render quick and easy
plots for your optimizer. It can plot the best cost for each iteration, and
show animations of the particles in 2-D and 3-D space. Furthermore, because
it has :code:`matplotlib` running under the hood, the plots are easily
customizable.

For example, if we want to plot the cost, simply run the optimizer, get the
cost history from the optimizer instance, and pass it to the
:code:`plot_cost_history()` method

.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions.single_obj import sphere
    from pyswarms.utils.plotters import plot_cost_history

    # Set up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Obtain cost history from optimizer instance
    cost_history = optimizer.cost_history

    # Plot!
    plot_cost_history(cost_history)
    plt.show()

In case you want to plot the particle movement, it is important that either
one of the :code:`matplotlib` animation :code:`Writers` is installed. These
doesn't come out of the box for :code:`pyswarms`, and must be installed
separately. For example, in a Linux or Windows distribution, you can install
:code:`ffmpeg` as

    >>> conda install -c conda-forge ffmpeg

Now, if you want to plot your particles in a 2-D environment, simply pass
the position history of your swarm (obtainable from swarm instance):


.. code-block:: python

    import pyswarms as ps
    from pyswarms.utils.functions.single_obj import sphere
    from pyswarms.utils.plotters import plot_cost_history

    # Set up optimizer
    options = {'c1':0.5, 'c2':0.3, 'w':0.9}
    optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2,
                                        options=options)

    # Obtain pos history from optimizer instance
    pos_history = optimizer.pos_history

    # Plot!
    plot_contour(pos_history)

You can also supply various arguments in this method: the indices of the
specific dimensions to be used, the limits of the axes, and the interval/
speed of animation.
"""

# Import standard library
import logging

# Import modules
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
from matplotlib import animation, cm
from mpl_toolkits.mplot3d import Axes3D


def plot_cost_history(
    cost_history, ax=None, title="Cost History", designer=None, **kwargs
):
    """Create a simple line plot with the cost in the y-axis and
    the iteration at the x-axis

    Parameters
    ----------
    cost_history : array_like
        Cost history of shape :code:`(iters, )` or length :code:`iters` where
        each element contains the cost for the given iteration.
    ax : :obj:`matplotlib.axes.Axes`, optional
        The axes where the plot is to be drawn. If :code:`None` is
        passed, then the plot will be drawn to a new set of axes.
    title : str, optional
        The title of the plotted graph. Default is `Cost History`
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :class:`matplotlib.axes.Axes`

    Returns
    -------
    :obj:`matplotlib.axes._subplots.AxesSubplot`
        The axes on which the plot was drawn.
    """
    try:
        # Infer number of iterations based on the length
        # of the passed array
        iters = len(cost_history)

        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(legend="Cost", label=["Iterations", "Cost"])

        # If no ax supplied, create new instance
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=designer.figsize)

        # Plot with iters in x-axis and the cost in y-axis
        ax.plot(np.arange(iters), cost_history, "k", lw=2, label=designer.legend)

        # Customize plot depending on parameters
        ax.set_title(title, fontsize=designer.title_fontsize)
        ax.legend(fontsize=designer.text_fontsize)
        ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
        ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
        ax.tick_params(labelsize=designer.text_fontsize)
    except TypeError:
        # logger.exception("Please check your input type")
        raise
    else:
        return ax


def plot_contour(
    pos_history,
    canvas=None,
    title="Trajectory",
    mark=None,
    designer=None,
    mesher=None,
    animator=None,
    n_processes=None,
    **kwargs
):
    """Draw a 2D contour map for particle trajectories

    Here, the space is represented as a flat plane. The contours indicate the
    elevation with respect to the objective function. This works best with
    2-dimensional swarms with their fitness in z-space.

    Parameters
    ----------
    pos_history : numpy.ndarray or list
        Position history of the swarm with shape
        :code:`(iteration, n_particles, dimensions)`
    canvas : (:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`),
        The (figure, axis) where all the events will be draw. If :code:`None`
        is supplied, then plot will be drawn to a fresh set of canvas.
    title : str, optional
        The title of the plotted graph. Default is `Trajectory`
    mark : tuple, optional
        Marks a particular point with a red crossmark. Useful for marking
        the optima.
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    mesher : :obj:`pyswarms.utils.formatters.Mesher`, optional
        Mesher class for mesh plots
    animator : :obj:`pyswarms.utils.formatters.Animator`, optional
        Animator class for custom animation
    n_processes : int
        number of processes to use for parallel mesh point calculation (default: None = no parallelization)
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :obj:`matplotlib.axes.Axes` plotting function

    Returns
    -------
    :obj:`matplotlib.animation.FuncAnimation`
        The drawn animation that can be saved to mp4 or other
        third-party tools
    """

    try:
        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(limits=[(-1, 1), (-1, 1)], label=["x-axis", "y-axis"])

        # If no Animator class supplied, use defaults
        if animator is None:
            animator = Animator()

        # If ax is default, then create new plot. Set-up the figure, the
        # axis, and the plot element that we want to animate
        if canvas is None:
            fig, ax = plt.subplots(1, 1, figsize=designer.figsize)
        else:
            fig, ax = canvas

        frame_text = ax.text(
            0.05,
            0.95,
            s="",
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
        )

        # Get number of iterations
        n_iters = len(pos_history)

        # Customize plot
        ax.set_title(title, fontsize=designer.title_fontsize)
        ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
        ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
        ax.set_xlim(designer.limits[0])
        ax.set_ylim(designer.limits[1])

        # Make a contour map if possible
        if mesher is not None:
            (xx, yy, zz) = _mesh(mesher, n_processes=n_processes)
            ax.contour(xx, yy, zz, levels=mesher.levels)

        # Mark global best if possible
        if mark is not None:
            ax.scatter(mark[0], mark[1], color="red", marker="x")

        # Put scatter skeleton
        plot = ax.scatter(x=[], y=[], c="black", alpha=0.6, **kwargs)

        # Do animation
        anim = animation.FuncAnimation(
            fig=fig,
            func=_animate,
            frames=range(n_iters),
            fargs=(pos_history, plot),
            interval=animator.interval,
            repeat=animator.repeat,
            repeat_delay=animator.repeat_delay,
        )
    except TypeError:
        print("Please check your input type")
        # rep.logger.exception("Please check your input type")
        raise
    else:
        return anim


def plot_surface(
    pos_history,
    canvas=None,
    title="Trajectory",
    designer=None,
    mesher=None,
    animator=None,
    mark=None,
    n_processes=None,
    **kwargs
):
    """Plot a swarm's trajectory in 3D

    This is useful for plotting the swarm's 2-dimensional position with
    respect to the objective function. The value in the z-axis is the fitness
    of the 2D particle when passed to the objective function. When preparing the
    position history, make sure that the:

    * first column is the position in the x-axis,
    * second column is the position in the y-axis; and
    * third column is the fitness of the 2D particle

    The :class:`pyswarms.utils.plotters.formatters.Mesher` class provides a
    method that prepares this history given a 2D pos history from any
    optimizer.

    .. code-block:: python

        import pyswarms as ps
        from pyswarms.utils.functions.single_obj import sphere
        from pyswarms.utils.plotters import plot_surface
        from pyswarms.utils.plotters.formatters import Mesher

        # Run optimizer
        options = {'c1':0.5, 'c2':0.3, 'w':0.9}
        optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=2, options)

        # Prepare position history
        m = Mesher(func=sphere)
        pos_history_3d = m.compute_history_3d(optimizer.pos_history)

        # Plot!
        plot_surface(pos_history_3d)

    Parameters
    ----------
    pos_history : numpy.ndarray
        Position history of the swarm with shape
        :code:`(iteration, n_particles, 3)`
    objective_func : callable
        The objective function that takes a swarm of shape
        :code:`(n_particles, 2)` and returns a fitness array
        of :code:`(n_particles, )`
    canvas : (:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`),
        The (figure, axis) where all the events will be draw. If :code:`None`
        is supplied, then plot will be drawn to a fresh set of canvas.
    title : str, optional
        The title of the plotted graph. Default is `Trajectory`
    mark : tuple, optional
        Marks a particular point with a red crossmark. Useful for marking the
        optima.
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    mesher : :obj:`pyswarms.utils.formatters.Mesher`, optional
        Mesher class for mesh plots
    animator : :obj:`pyswarms.utils.formatters.Animator`, optional
        Animator class for custom animation
    n_processes : int
        number of processes to use for parallel mesh point calculation (default: None = no parallelization)
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :class:`matplotlib.axes.Axes` plotting function

    Returns
    -------
    :class:`matplotlib.animation.FuncAnimation`
        The drawn animation that can be saved to mp4 or other
        third-party tools
    """
    try:
        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(
                limits=[(-1, 1), (-1, 1), (-1, 1)],
                label=["x-axis", "y-axis", "z-axis"],
                colormap=cm.viridis,
            )

        # If no Animator class supplied, use defaults
        if animator is None:
            animator = Animator()

        # Get number of iterations
        # If ax is default, then create new plot. Set-up the figure, the
        # axis, and the plot element that we want to animate
        if canvas is None:
            fig = plt.figure(figsize=designer.figsize)
        else:
            fig, ax = canvas

        # Initialize 3D-axis
        ax = plt.axes(projection="3d")  # Axes3D(fig)
        ax.grid()

        frame_text = ax.text(
            0.05 * designer.limits[0][0],
            0.95 * designer.limits[1][1],
            z=0.95 * designer.limits[2][1],
            s="",
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
        )

        n_iters = len(pos_history)

        # Customize plot
        ax.set_title(title, fontsize=designer.title_fontsize)
        ax.set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
        ax.set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
        ax.set_zlabel(designer.label[2], fontsize=designer.text_fontsize)
        ax.set_xlim(designer.limits[0])
        ax.set_ylim(designer.limits[1])
        ax.set_zlim(designer.limits[2])

        # Make a contour map if possible
        if mesher is not None:
            (xx, yy, zz) = _mesh(mesher, n_processes=n_processes)
            ax.plot_surface(xx, yy, zz, cmap=designer.colormap, alpha=mesher.alpha)

        # Mark global best if possible
        if mark is not None:
            ax.scatter(mark[0], mark[1], mark[2], color="red", marker="x")

        # Put scatter skeleton
        plot = ax.scatter(xs=[], ys=[], zs=[], c="black", alpha=0.6, **kwargs)

        # Do animation
        anim = animation.FuncAnimation(
            fig=fig,
            func=_animate,
            frames=range(n_iters),
            fargs=(pos_history, plot),
            interval=animator.interval,
            repeat=animator.repeat,
            repeat_delay=animator.repeat_delay,
        )
    except TypeError:
        print("Please check your input type")
        raise
    else:
        return anim


def plot_summary(
    optimizers,
    canvas=None,
    title="Trajectory",
    titles=None,
    mark=None,
    designer=None,
    mesher=None,
    animator=None,
    n_processes=None,
    **kwargs
):
    """Draw a 2D contour map for particle trajectories

    Here, the space is represented as a flat plane. The contours indicate the
    elevation with respect to the objective function. This works best with
    2-dimensional swarms with their fitness in z-space.

    Parameters
    ----------
    optimizers : numpy.ndarray or list
        List of optimizations to summarize
        :code:`(iteration, n_particles, dimensions)`
    canvas : (:obj:`matplotlib.figure.Figure`, :obj:`matplotlib.axes.Axes`),
        The (figure, axis) where all the events will be draw. If :code:`None`
        is supplied, then plot will be drawn to a fresh set of canvas.
    title : str, optional
        The title of the plotted graph. Default is `Trajectory`
    mark : tuple, optional
        Marks a particular point with a red crossmark. Useful for marking
        the optima.
    designer : :obj:`pyswarms.utils.formatters.Designer`, optional
        Designer class for custom attributes
    mesher : :obj:`pyswarms.utils.formatters.Mesher`, optional
        Mesher class for mesh plots
    animator : :obj:`pyswarms.utils.formatters.Animator`, optional
        Animator class for custom animation
    n_processes : int
        number of processes to use for parallel mesh point calculation (default: None = no parallelization)
    **kwargs : dict
        Keyword arguments that are passed as a keyword argument to
        :obj:`matplotlib.axes.Axes` plotting function

    Returns
    -------
    :obj:`matplotlib.animation.FuncAnimation`
        The drawn animation that can be saved to mp4 or other
        third-party tools
    """

    try:
        # If no Designer class supplied, use defaults
        if designer is None:
            designer = Designer(limits=[(-1, 1), (-1, 1)], label=["x-axis", "y-axis"])

        # If no Animator class supplied, use defaults
        if animator is None:
            animator = Animator()

        # If ax is default, then create new plot. Set-up the figure, the
        # axis, and the plot element that we want to animate
        if canvas is None:
            fig, ax = plt.subplots(3, len(optimizers), figsize=designer.figsize)
        else:
            fig, ax = canvas

        frame_text = ax[2, 0].text(
            0.05,
            0.95,
            s="",
            transform=ax[0, 0].transAxes,
            horizontalalignment="left",
            verticalalignment="top",
        )

        # Get number of iterations
        n_iters = len(optimizers[0].record_value["X"])

        # Customize plot
        fig.suptitle(title, fontsize=designer.title_fontsize)

        pos_histories = []
        plots = []

        for i, opt in enumerate(optimizers):
            # assert len(opt.record_value['X']) == len(optimizers[0].record_value['X'])
            if titles:
                ax[0, i].set_title(titles[i])

            Y_history = pd.DataFrame(
                np.array(opt.record_value["Y"]).reshape((-1, opt.size_pop))
            )
            ax[1, i].set_title(
                str(opt.gbest_y) + " @ X: " + str(opt.gbest_x), fontsize=8
            )
            ax[0, i].plot(Y_history.index, Y_history.values, ".")
            Y_history.min(axis=1).cummin().plot(kind="line", ax=ax[1, i])

            ax[2, i].set_xlabel(designer.label[0], fontsize=designer.text_fontsize)
            ax[2, i].set_ylabel(designer.label[1], fontsize=designer.text_fontsize)
            ax[2, i].set_xlim(designer.limits[0])
            ax[2, i].set_ylim(designer.limits[1])

            # Make a contour map if possible
            if mesher is not None:
                (xx, yy, zz) = _mesh(mesher, n_processes=n_processes)
                ax[2, i].contour(xx, yy, zz, levels=mesher.levels)

            # Mark global best if possible
            if mark is not None:
                ax[2, i].scatter(mark[0], mark[1], color="red", marker="x")

            # Put scatter skeleton
            plots.append(ax[2, i].scatter(x=[], y=[], c="black", alpha=0.6, **kwargs))
            pos_histories.append(np.asarray(opt.record_value["X"]))

        # Do animation
        anim = animation.FuncAnimation(
            fig=fig,
            func=_animate_summary,
            frames=range(n_iters),
            fargs=(pos_histories, plots),
            interval=animator.interval,
            repeat=animator.repeat,
            repeat_delay=animator.repeat_delay,
        )
    except TypeError:
        print("Please check your input type")
        raise
    else:
        return anim


def _animate(i, data, plot):
    """Helper animation function that is called sequentially
    :class:`matplotlib.animation.FuncAnimation`
    """
    current_pos = data[i]
    if i % 10 == 0:
        plot.axes.texts[0].set_text(str(i))

    if np.array(current_pos).shape[1] == 2:
        plot.set_offsets(current_pos)
    else:
        plot._offsets3d = current_pos.T
    return (plot,)


def _animate_summary(i, data, plots):
    """Helper animation function that is called sequentially
    IT ACTUALLY WORKS
    :class:`matplotlib.animation.FuncAnimation`
    """
    if i % 10 == 0:
        plots[0].axes.texts[0].set_text(str(i))

    for j, plot in enumerate(plots):
        opt_data = data[j]
        current_pos = opt_data[i] if i < len(opt_data) else opt_data[-1]

        if np.array(current_pos).shape[1] == 2:
            plot.set_offsets(current_pos)
        else:
            plot._offsets3d = current_pos.T
    return (plots,)


def _mesh(mesher, n_processes=None):
    """Helper function to make a mesh"""
    xlim = mesher.limits[0]
    ylim = mesher.limits[1]
    x = np.arange(xlim[0], xlim[1], mesher.delta)
    y = np.arange(ylim[0], ylim[1], mesher.delta)
    xx, yy = np.meshgrid(x, y)
    xypairs = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T

    # Get z-value

    # Setup Pool of processes for parallel evaluation
    pool = None if n_processes is None else mp.Pool(n_processes)

    if pool is None:
        z = mesher.func(xypairs)
    else:
        results = pool.map(mesher.func, np.array_split(xypairs, pool._processes))
        z = np.concatenate(results)

    # Close Pool of Processes
    if n_processes is not None:
        pool.close()

    zz = z.reshape(xx.shape)
    return (xx, yy, zz)
