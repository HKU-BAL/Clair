"""
author: Chi Ian Tang
version: 1.2 beta
date: 2018 - 06 - 14
"""

from __future__ import print_function, division
import sys
sys.path.append('../')
import numpy as np
import time
import os, errno
import re
import argparse
import pickle
from random import randrange
from multiprocessing import Process, Pool
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

VERSION = "1.1"

def split_list(long_list, wanted_parts=1):
    length = len(long_list)
    return [ long_list[i*length // wanted_parts: (i+1)*length // wanted_parts] for i in range(wanted_parts) ]

def load_tensor(file_name, all_components = False):
    """
    Load the tensor in binary format.

    all_components (boolean): return all components of the pickled file (total, X, Y, pos) if True, otherwise only return (total, X)
    """
    import clairvoyante.utils_v2 as utils
    with open(file_name, "rb") as fh:
        if all_components:
            total = pickle.load(fh)
            XArrayCompressed = pickle.load(fh)
            XArray, _, _ = utils.DecompressArray(XArrayCompressed, 0, total, total)
            del XArrayCompressed

            YArrayCompressed = pickle.load(fh)
            YArray, _, _ = utils.DecompressArray(YArrayCompressed, 0, total, total)
            del YArrayCompressed

            posArrayCompressed = pickle.load(fh)
            posArray, _, _ = utils.DecompressArray(posArrayCompressed, 0, total, total)
            del posArrayCompressed

            return total, XArray, YArray, posArray
        else:
            total = pickle.load(fh)
            XArrayCompressed = pickle.load(fh)
            XArray, _, _ = utils.DecompressArray(XArrayCompressed, 0, total, total)
            del XArrayCompressed

            YArrayCompressed = pickle.load(fh)
            YArray, _, _ = utils.DecompressArray(YArrayCompressed, 0, total, total)
            del YArrayCompressed
            
            return total, XArray, YArray

def build_position_array_file(file_name, posArray):
    """
    Cache the position array into a text file (newline separated format).
    """

    with open(file_name, "w+") as f:
        for element in posArray:
            f.write(element + "\n")

def load_position_array(file_name):
    """
    Read the position array from a text file (tab or newline separated) and build a lookup dictionary.
    """
    with open(file_name, "r") as f:
        full_file = f.read()
        splitted = re.split("(?:\r\n|\n|\t)", full_file) # Compatible with Windows / Linux endline and Tab
        pos_lookup_dict = dict([(element, counter) for counter, element in enumerate(splitted)]) # Make lookup dictionary
    return pos_lookup_dict

def load_input_index_file(file_name):
    """
    Load the positions to be read, from a text file (tab or newline separated). (e.g. chr21:45053245\tchr21:45053245)
    """
    with open(file_name, "r") as f:
        full_file = f.read()
        read_positions = re.split("(?:\r\n|\n|\t)", full_file) # Compatible with Windows / Linux endline and Tab
    return read_positions

def draw_positions_async(XArray, YArray, pos_index_list, max_reference=50, max_relative=50, width=15, height=9, workers=1, worker_id=None, x_only=False, output_dir=".", pool=False, dpi=None, file_format='png'):
    """
    Plot the matrices into a image file.
    
    workers (int): when workers == 1, it runs in a single-threaded mode, otherwise, it spawns the same number of workers, each running draw_positions with workers=1
    """
    if workers == 1:
        plot_num_row = 5
        if x_only:
            # Only 4 Matrices
            plot_num_row = 4
        else:
            # 4 + Table
            plot_num_row = 5

        col_labels = ['A', 'C', 'G', 'T', 'HET', 'HOM', 'REF', 'SNP', 'INS', 'DEL', '0', '1', '2', '3', '4', '>4']
        # Init
        init_mat = XArray[0,:,:,0].transpose()
        x_ticker = mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True, symmetric=True, prune='both')
        x_max = init_mat.shape[1]
        x_tick_values = list(map(lambda x : x + (x_max - 1) // 2, x_ticker.tick_values((-1) * (x_max - 1) // 2, (x_max - 1) // 2))) + [0, x_max - 1]

        save_threads = []
        def save_figure(figure, filename, file_format):
            figure.savefig(filename, bbox_inches='tight', dpi=dpi, format=file_format)
            plt.close(figure)
            del figure

        try:
            for i, pos_index in enumerate(pos_index_list): # for each position to be read
                position, index = pos_index
                if worker_id is not None:
                    print("Worker", worker_id, "Drawing", i + 1, repr(position), index)
                else:
                    print("Drawing", i + 1, repr(position), index)
                
                f = plt.figure(figsize=(width, height))
                plt.suptitle(position, fontsize=width*2)
                plt.axis('off')
                ax = f.add_subplot(plot_num_row, 1, 1)
                im = ax.matshow(XArray[index,:,:,0].transpose(), vmin=0, vmax=max_reference, cmap=plt.cm.hot)
                ax.xaxis.set_ticks(x_tick_values)
                ax.yaxis.set_ticks([])
                f.colorbar(im, ax=ax, ticks=mticker.MaxNLocator(nbins=5))

                for i in range(1, 4):
                    ax = f.add_subplot(plot_num_row, 1, i + 1)
                    im = ax.matshow(XArray[index,:,:,i].transpose(), vmin=-max_relative, vmax=max_relative, cmap=plt.cm.bwr)
                    ax.xaxis.set_ticks(x_tick_values)
                    ax.yaxis.set_ticks([])
                    f.colorbar(im, ax=ax, ticks=mticker.MaxNLocator(nbins=5))

                if not x_only:
                    f.add_subplot(plot_num_row, 1, 5)
                    plt.axis('off')
                    col_labels = ['A', 'C', 'G', 'T', 'HET', 'HOM', 'REF', 'SNP', 'INS', 'DEL', '0', '1', '2', '3', '4', '>4']
                    table_vals = YArray[index, :]
                    cell_text = [['%1.2f' % x for x in table_vals]]
                    table = plt.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
                    table.scale(1, 2)

                
                p = Process(target=save_figure, args=(f, output_dir + "/" + position + "." + file_format, file_format,))
                p.start()
                save_threads.append(p)
                # plt.savefig(position + ".png", bbox_inches='tight', dpi=dpi)
                plt.close(f)
                del f
            
            for p in save_threads:
                p.join()
            if worker_id is not None:
                print("Worker", worker_id, "Save threads finshed")
            else:
                print("Save threads finshed")
        except KeyboardInterrupt:
            for p in save_threads:
                p.terminate()
                p.join()
            print("All save threads Terminated")
            raise KeyboardInterrupt

    else:
        # Multithreading Mode

        if pool:
            # Using Python Pool Multithreading (slower for small w)
            splitted = split_list(pos_index_list, workers)
            pool = Pool(processes=workers)
            try:
                for i in range(workers):
                    
                    pool.apply_async(draw_positions_async, args=(XArray, YArray, splitted[i],), 
                        kwds={'max_reference':max_reference, 'max_relative':max_relative, 'width':width, 'height':height, 'workers':1, 'worker_id':i, 
                        'x_only':x_only, 'output_dir':output_dir, 'dpi':dpi, 'file_format':file_format})
                    print("Worker", i, "Drawing", len(splitted[i]), "Plots")
                pool.close()
                pool.join()
                print("All Workers Finished")
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                print("All Workers Terminated")
        else:
            # Split into equal pieces
            splitted = split_list(pos_index_list, workers)
            processes = []
            try:
                # Create Threads
                for i in range(workers):
                    p = Process(target=draw_positions_async, args=(XArray, YArray, splitted[i],), 
                        kwargs={'max_reference':max_reference, 'max_relative':max_relative, 'width':width, 'height':height, 'workers':1, 'worker_id':i, 
                        'x_only':x_only, 'output_dir':output_dir, 'dpi':dpi, 'file_format':file_format})
                    print("Worker", i, p, "Drawing", len(splitted[i]), "Plots")
                    p.start()
                    processes.append(p)

                # Wait for threads to end
                for i, p in enumerate(processes):
                    p.join()
                    print("Worker", i, p, "Finished")
            except KeyboardInterrupt:
                # Terminate Threads when interrupt is received
                for i, p in enumerate(processes):
                    p.terminate()
                    p.join()
                    print("Worker", i, p, "Terminated")
                print("All Workers Terminated")



def draw_positions_sync(XArray, YArray, pos_index_list, 
        max_reference=50, max_relative=50, width=15, height=9, workers=1, worker_id=None, x_only=False, output_dir=".", pool=False, dpi=None, file_format='png'):
    """
    Plot the matrices into a PNG file.
    
    workers (int): when workers == 1, it runs in a single-threaded mode, otherwise, it spawns the same number of workers, each running draw_positions with workers=1
    """
    if workers == 1:
        plot_num_row = 5
        if x_only:
            # Only 4 Matrices
            plot_num_row = 4
        else:
            # 4 + Table
            plot_num_row = 5

        col_labels = ['A', 'C', 'G', 'T', 'HET', 'HOM', 'REF', 'SNP', 'INS', 'DEL', '0', '1', '2', '3', '4', '>4']
        # Init
        subplots = []
        images = []
        f = plt.figure(figsize=(width, height))
        f.suptitle("Init", fontsize=width*2)
        plt.axis('off')

        init_mat = XArray[0,:,:,0].transpose()
        ax = f.add_subplot(plot_num_row, 1, 1)
        ax.xaxis.tick_top()
        ax.xaxis.set_ticks_position('both')
        x_ticker = mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True, symmetric=True, prune='both')
        x_max = init_mat.shape[1]
        x_tick_values = list(map(lambda x : x + (x_max - 1) // 2, x_ticker.tick_values((-1) * (x_max - 1) // 2, (x_max - 1) // 2))) + [0, x_max - 1]
        ax.xaxis.set_ticks(x_tick_values)
        # ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 4, 5, 10], integer=True))
        ax.yaxis.set_ticks([])
        # ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        
        im = ax.imshow(init_mat, vmin=0, vmax=max_reference, cmap=plt.cm.hot, origin='upper', interpolation='nearest', aspect='equal')
        f.colorbar(im, ax=ax, ticks=mticker.MaxNLocator(nbins=5))
        subplots.append(ax)
        images.append(im)
        for i in range(2, 5):
            ax = f.add_subplot(plot_num_row, 1, i)
            im = ax.imshow(init_mat, vmin=-max_relative, vmax=max_relative, cmap=plt.cm.bwr, origin='upper', interpolation='nearest', aspect='equal')
            ax.xaxis.tick_top()
            ax.xaxis.set_ticks_position('both')
            # ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 4, 5, 6, 10], integer=True, symmetric=True))
            ax.xaxis.set_ticks(x_tick_values)
            ax.yaxis.set_ticks([])
            # ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
            f.colorbar(im, ax=ax, ticks=mticker.MaxNLocator(nbins=5))
            subplots.append(ax)
            images.append(im)

        if not x_only:
            ax = f.add_subplot(plot_num_row, 1, 5)
            ax.axis('off')
            col_labels = ['A', 'C', 'G', 'T', 'HET', 'HOM', 'REF', 'SNP', 'INS', 'DEL', '0', '1', '2', '3', '4', '>4']
            table_vals = YArray[0, :]
            cell_text = [['%1.2f' % x for x in table_vals]]
            table = ax.table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
            table.scale(1, 2)
            subplots.append(ax)
        
        for i, pos_index in enumerate(pos_index_list): # for each position to be read
            position, index = pos_index

            if worker_id is not None:
                print("Worker", worker_id, "Drawing", i + 1, repr(position), index)
            else:
                print("Drawing", i + 1, repr(position), index)

            f.suptitle(position, fontsize=width*2)
            for i in range(4):
                images[i].set_data(XArray[index,:,:,i].transpose())

            if not x_only:
                subplots[4].cla()
                subplots[4].axis('off')
                
                table_vals = YArray[index, :]
                cell_text = [['%1.2f' % x for x in table_vals]]
                table = subplots[4].table(cellText=cell_text, colLabels=col_labels, loc='center', cellLoc='center')
                table.scale(1, 2)
            f.canvas.draw()
            plt.draw()
            f.savefig(output_dir + "/" + position + "." + file_format, file_format=file_format, bbox_inches='tight', dpi=dpi)

        plt.close(f)
        del f
                
    else:
        # Multithreading Mode

        if pool:
            # Using Python Pool Multithreading (slower for small w)
            splitted = split_list(pos_index_list, workers)
            pool = Pool(processes=workers)
            try:
                for i in range(workers):
                    pool.apply_async(draw_positions_sync, args=(XArray, YArray, splitted[i],), 
                        kwds={'max_reference':max_reference, 'max_relative':max_relative, 'width':width, 'height':height, 'workers':1, 'worker_id':i, 
                        'x_only':x_only, 'output_dir':output_dir, 'dpi':dpi, 'file_format':file_format})
                    print("Worker", i, "Drawing", len(splitted[i]), "Plots")
                pool.close()
                pool.join()
                print("All Workers Finished")
            except KeyboardInterrupt:
                pool.terminate()
                pool.join()
                print("All Workers Terminated")
        
        else:

            # Split into equal pieces
            splitted = split_list(pos_index_list, workers)
            processes = []
            try:
                # Create Threads
                for i in range(workers):
                    p = Process(target=draw_positions_sync, args=(XArray, YArray, splitted[i],), 
                        kwargs={'max_reference':max_reference, 'max_relative':max_relative, 'width':width, 'height':height, 'workers':1, 'worker_id':i, 
                        'x_only':x_only, 'output_dir':output_dir, 'dpi':dpi, 'file_format':file_format})
                    print("Worker", i, p, "Drawing", len(splitted[i]), "Plots")
                    p.start()
                    processes.append(p)

                # Wait for threads to end
                for i, p in enumerate(processes):
                    p.join()
                    print("Worker", i, p, "Finished")
            except KeyboardInterrupt:
                # Terminate Threads when interrupt is received
                for i, p in enumerate(processes):
                    p.terminate()
                    p.join()
                    print("Worker", i, p, "Terminated")
                print("All Workers Terminated")

def draw_positions_plotly(XArray, YArray, pos_index_list, 
        max_reference=50, max_relative=50, width=15, height=9, workers=1, worker_id=None, x_only=False, output_dir=".", pool=False, dpi=None, file_format='png'):
    import plotly
    import plotly.graph_objs as go
    from plotly import tools
    reference_ticks = np.linspace(0.0, max_reference, num=11)
    reference_ticks_text = ["%.2f" % tick for tick in reference_ticks]
    relative_ticks = np.linspace(-max_relative, max_relative, num=11)
    relative_ticks_text = ["%.2f" % tick for tick in relative_ticks]

    x_domains = [[0, 0.9], [0.95, 1]]
    
    plot_num_row = 4
    y_domain = [[0.8, 1.0], [0.55, 0.75], [0.3, 0.5], [0, 0.2]]

    all_traces = []
    all_scales = []

    layout = {}
    for i in range(plot_num_row):
        layout['xaxis' + str(i * 2 + 1)] = dict(autotick=False, ticks='outside', side='top', tick0=0, dtick=1, tickcolor='#000', domain=x_domains[0])
        layout['yaxis' + str(i * 2 + 1)] = dict(autotick=False, visible=False, domain=y_domain[i])

        
        layout['xaxis' + str(i * 2 + 2)] = dict(autotick=False, visible=False, domain=x_domains[1])
        if i == 0:
            layout['yaxis' + str(i * 2 + 2)] = dict(side='right', ticks="", showticklabels=True, ticktext=reference_ticks_text, tickvals=[i for i in range(len(reference_ticks))], domain=y_domain[i])
        else:
            layout['yaxis' + str(i * 2 + 2)] = dict(side='right', ticks="", showticklabels=True, ticktext=relative_ticks_text, tickvals=[i for i in range(len(relative_ticks))], domain=y_domain[i])

    for i, pos_index in enumerate(pos_index_list): # for each position to be read
        position, index = pos_index
        if worker_id is not None:
            print("Worker", worker_id, "Drawing", i + 1, repr(position), index)
        else:
            print("Drawing", i + 1, repr(position), index)
        
        trace = go.Heatmap(z=np.flip(XArray[index,:,:,0].transpose(), 0),
                zmin=0, zmax=max_reference,
                colorscale="Hot", showscale=False, name=position + "reference", visible=False)
        
        scale = go.Heatmap(z=np.array([reference_ticks]).transpose(),
                zmin=0, zmax=max_reference,
                colorscale="Hot", showscale=False, name="reference scale", visible=False)
        all_traces.append(trace)
        all_scales.append(scale)

        for i in range(1, 4):
            trace = go.Heatmap(z=np.flip(XArray[index,:,:,i].transpose(),0),
                    zmin=-max_relative, zmax=max_relative,
                    colorscale="RdBu", showscale=False, visible=False, name=position + "relative")
            scale = go.Heatmap(z=np.array([relative_ticks]).transpose(),
                zmin=-max_relative, zmax=max_relative,
                colorscale="RdBu", showscale=False, visible=False, name="reference scale")
            all_traces.append(trace)
            all_scales.append(scale)
    
    steps = []
    for i, pos_index in enumerate(pos_index_list):
        position, index = pos_index
        step = dict(
            method = 'restyle',
            args = ['visible', [False] * (plot_num_row * len(pos_index_list) * 2)],
            label = position
        )
        for j in range(plot_num_row):
            step['args'][1][i * plot_num_row + j] = True
            step['args'][1][i * plot_num_row + j + len(pos_index_list) * plot_num_row] = True
        steps.append(step)

    sliders = [dict(
        active = 0,
        yanchor='top',
        currentvalue = {
            'font': {'size': 30},
            'prefix': '',
            'visible': True,
            'xanchor': 'center'
            },
        pad = {"t": 50},
        steps = steps
    )]

    layout['sliders'] = sliders

    fig = tools.make_subplots(rows=plot_num_row, cols=2)

    for i, t in enumerate(all_traces):
        fig.append_trace(t, i % plot_num_row + 1, 1)

    for i, s in enumerate(all_scales):
        fig.append_trace(s, i % plot_num_row + 1, 2)

    fig['layout'].update(layout)
    plotly.offline.plot(fig, filename='p.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Draw visualizations at positions" )

    parser.add_argument('-u', '--update', action='store_true',
            help="Force update position cache file")
    
    parser.add_argument('-g', '--gen_pos_only', action='store_true',
            help="Generate position cache file only")

    parser.add_argument('-u', '--update', action='store_true',
            help="Force update position cache file")

    parser.add_argument('-f', '--read_file', type=str, default="readPositionList.txt",
            help="Position file,  default: %(default)s")

    parser.add_argument('-a', '--all', action='store_true',
            help="Render all positions, ignoring input file (-f flag)")

    parser.add_argument('-o', '--output_dir', type=str, default=".",
            help="The output directory,  default: %(default)s")
    
    parser.add_argument('-w', '--workers', type=int, default=8,
            help="Number of workers,  default: %(default)s")

    parser.add_argument('-x', '--x_only', action='store_true',
            help="Draw matrices only")

    parser.add_argument('-r', '--resume', action='store_true',
            help="Resume plotting the remaining positions")
    
    parser.add_argument('-m', '--max_plot', type=int, default=-1,
            help="Maximum number of plots, negative number means no upper bound,  default: %(default)s")

    parser.add_argument('-p', '--plotly', action='store_true',
            help="Plot into a plotly file")

    parser.add_argument('--cached', action='store_true',
            help="X array and Y array caching")

    parser.add_argument('--size', type=str, default="15-9",
            help="Figure size,  default: %(default)s")

    parser.add_argument('--dpi', type=int, default=None,
            help="Plot dpi, default: %(default)s")

    parser.add_argument('--format', type=str, default='png',
            help="Plot file format. Support png, pdf, ps, eps and svg. default: %(default)s")
    
    parser.add_argument('--max_ref', type=int, default=50,
            help="Maximum reference value for plotting,  default: %(default)s")

    parser.add_argument('--max_rel', type=int, default=50,
            help="Maximum relative value for plotting,  default: %(default)s")

    parser.add_argument('--async_save', action='store_true',
            help="Asynchronized saving mode")
    
    parser.add_argument('--pool', action='store_true',
            help="Python multiprocessing Pool mode (Experimental)")

    parser.add_argument('--version', action='store_true',
            help="Print script version and exit")

            
    args = parser.parse_args()

    if args.version:
        print("MatPlot, version:" + VERSION)
        sys.exit(0)

    try:
        os.makedirs(args.output_dir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    
    if not os.path.isfile("posArray.txt") or args.update or args.gen_pos_only or args.all:
        # Generate the position dictionary cache file if necessary
        total, XArray, YArray, posArray = load_tensor(args.tensor_file, all_components = True)
        print("Total:", total)
        print("Building Position Array File")
        build_position_array_file("posArray.txt", posArray)
        if args.gen_pos_only:
            print("Generated Position Array File")
            sys.exit(0)
    elif args.cached:
        if not os.path.isfile("XArray.npy") or not os.path.isfile("YArray.npy"):
            total, XArray, YArray = load_tensor(args.tensor_file, all_components = False)
            np.save("XArray", XArray)
            np.save("YArray", YArray)
        else:
            XArray = np.load("XArray.npy")
            YArray = np.load("YArray.npy")
    else:
        total, XArray, YArray = load_tensor(args.tensor_file, all_components = False)

    pos_lookup_dict = load_position_array("posArray.txt")
    if args.all:
        read_positions = load_input_index_file("posArray.txt")
    else:
        read_positions = load_input_index_file(args.read_file)
    
    pos_index_list = list(filter(lambda pos_index: pos_index[1] is not None and len(pos_index[0]) > 0, map(lambda x: (x, pos_lookup_dict.get(x)), read_positions)))
    if args.resume and not args.plotly:
        all_files = os.listdir(args.output_dir + "/")
        pos_index_list = list(filter(lambda pos_index: (pos_index[0] + "." + args.format) not in all_files, pos_index_list))
    
    if args.max_plot >= 0:
        pos_index_list = pos_index_list[:args.max_plot]

    print("Drawing %d Positions:" % len(pos_index_list), pos_index_list[:10])
    width = int(args.size.split("-")[0])
    height = int(args.size.split("-")[1])
    if args.async_save:
        draw_positions_async(XArray, YArray, pos_index_list, 
            max_reference=args.max_ref, max_relative=args.max_rel, width=width, height=height, workers=args.workers, 
            x_only=args.x_only, output_dir=args.output_dir, pool=args.pool, dpi=args.dpi, file_format=args.format)
    elif args.plotly:
        pos_index_list = list(sorted(pos_index_list, key=lambda pos_index: pos_index[0]))
        draw_positions_plotly(XArray, YArray, pos_index_list, 
            max_reference=args.max_ref, max_relative=args.max_rel, width=width, height=height, workers=args.workers, 
            x_only=args.x_only, output_dir=args.output_dir, pool=args.pool, dpi=args.dpi, file_format=args.format)
    else:
        draw_positions_sync(XArray, YArray, pos_index_list, 
            max_reference=args.max_ref, max_relative=args.max_rel, width=width, height=height, workers=args.workers, 
            x_only=args.x_only, output_dir=args.output_dir, pool=args.pool, dpi=args.dpi, file_format=args.format)
    
    print("Finished Drawing")



