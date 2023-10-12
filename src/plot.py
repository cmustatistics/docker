#########################################
# Utilities for plotting experiments
#########################################

import pandas as pd
from matplotlib import pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
import tensorflow as tf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
import pathlib

#################
# Configuration
#################

RMSE_label = 'RMSE'
ITS_label = '# mini-batches'
LL_label = 'Avg. Log-Likelihood'
TIME_label = "time (seconds)"

TIME = 'total runtime'
RMSE = 'val rmse'
LL = 'val avg log-likelihood per point'
ITS = 'step'
LENGTHSCALE = 'lengthscale 0'
VARIANCE = 'variance'

LIA_simple_name="LIA",
LIA_to_PP_simple_name="LIA -> Prop",
SVGP_simple_name="LSVGP"

#################

def get_metric(metric, event_acc, its_cut=None, its_label=None):
    data = []

    for _, s, t in event_acc.Tensors(metric):
        data.append((s, tf.make_ndarray(t)))

    df = pd.DataFrame(data, columns=['step', 'tensor'])

    df = df.rename(columns={'tensor': metric})

    if its_cut is not None and its_label is not None:
        df = df[df[its_label] <= its_cut]

    return df


def merge(df1, df2):
    if df1 is None:
        return df2

    return pd.merge(df1, df2, on='step')


def add(df1, df2):
    if df1 is None:
        return df2

    return df1 + df2


def get_columns(metric, runs):
    return ["{0}_{1}".format(metric, r) for r in range(runs)]


def method(method_str, metrics, experiment, name, color,
           its_cut=None, its_label=None, simple_name=None, runs=1,
           base=None):
    if base is None:
        base = "../benchmark"

    method_path = "{0}/{1}/{2}".format(base,experiment, method_str)

    dfs = []
    dfavg = None
    for run in range(runs):

        run_path = Path("{0}/run_{1}/".format(method_path, run)).glob('*')
        run_path = str(next(run_path))

        event_acc = EventAccumulator(run_path, size_guidance={
            "compressed_histograms": 500,
            "images": 1,
            "audio": 1,
            "scalars": 1,
            "histograms": 1,
            "tensors": 0})
        event_acc.Reload()

        df_run = None
        for metric in metrics:
            df_run = merge(df_run,
                           get_metric(metric, event_acc, its_cut=its_cut,
                                      its_label=its_label))

        dfavg = add(dfavg, df_run)
        dfs.append(df_run)

    dfavg = dfavg / runs
    if runs == 1:
        df = dfs[0]
    else:
        df = dfavg
        for run in range(runs):
            df = pd.merge(df, dfs[run], on=ITS,
                          suffixes=[None, '_{0}'.format(run)])
        for metric in metrics:
            df[metric + "_std"] = df[get_columns(metric, runs)].std(axis=1)

    if simple_name is not None:
        df.simple_name = simple_name
    df.name = name
    df.color = color
    return df


def plot_methods(ax, methods, xname, yname, xlabel=None, ylabel=None, runs=1,
                 title=None, horizontal_bar_methods=None, fontsize=14,
                 labelpad=30):
    width = 2
    xmin, xmax = None, None

    for method in methods:
        xs = method[xname]
        xmin = xs.min() if xmin is None else min(xmin, xs.min())
        xmax = xs.max() if xmax is None else max(xmax, xs.max())
        ax.plot(xs, method[yname], label=method.name, c=method.color,
                linewidth=width)
        if runs > 1:
            for column in get_columns(yname, runs):
                ax.plot(xs, method[column], linewidth=width,
                        c=method.color, alpha=0.2)

    if horizontal_bar_methods is not None:
        for method in horizontal_bar_methods:
            ax.hlines(y=method[yname].values[-1],
                      label=method.name, linewidth=width,
                      xmin=xmin, xmax=xmax, colors=method.color)
            if runs > 1:
                for column in get_columns(yname, runs):
                    ax.hlines(y=method[column].values[-1], linewidth=width,
                              xmin=xmin, xmax=xmax, colors=method.color,
                              alpha=0.2)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=fontsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=fontsize, rotation=0, labelpad=labelpad)
    if title is not None:
        ax.set_title(title, fontsize=fontsize)


def load(dataset_str,
         metrics,
         its_cut=None,
         its_label=ITS,
         LIA_to_LIF_str=None,
         LIA_to_PP_str=None,
         LIF_str=None,
         LIA_str=None,
         SVGP_str=None,
         classic_metrics=None,
         GP_str=None,
         VFE_str=None,
         runs=1,
         base=None):
    methods = []

    if LIA_to_LIF_str is not None:
        LIA_to_LIF = \
            method(
                base=base,
                method_str=LIA_to_LIF_str,
                runs=runs,
                experiment=dataset_str,
                metrics=metrics,
                name=r'$\mathcal{L}_{IA} \rightarrow \mathcal{L}_{IF}$',
                color="cyan",
                its_label=its_label, its_cut=its_cut)
        methods.append(LIA_to_LIF)

    if LIA_to_PP_str is not None:
        LIA_to_LIF_no_grad = \
            method(
                base=base,
                runs=runs,
                simple_name=LIA_to_PP_simple_name,
                method_str=LIA_to_PP_str,
                experiment=dataset_str,
                metrics=metrics, name=r'$\mathcal{L}_{IA}\rightarrow$PP',
                color="red",
                its_label=its_label, its_cut=its_cut)
        methods.append(LIA_to_LIF_no_grad)

    if LIF_str is not None:
        LIF = \
            method(
                base=base,
                runs=runs,
                method_str=LIF_str,
                simple_name="LIF",
                experiment=dataset_str,
                metrics=metrics, name=r'$\mathcal{L}_{IF}$', color="b",
                its_label=its_label, its_cut=its_cut)
        methods.append(LIF)

    if LIA_str is not None:
        LIA = \
            method(
                base=base,
                runs=runs,
                method_str=LIA_str,
                experiment=dataset_str,
                simple_name=LIA_simple_name,
                metrics=metrics, name=r'$\mathcal{L}_{IA}$', color="k",
                its_label=its_label, its_cut=its_cut)
        methods.append(LIA)

    if SVGP_str is not None:
        SVI = \
            method(method_str=SVGP_str,
                   base=base,
                   runs=runs,
                   experiment=dataset_str,
                   simple_name=SVGP_simple_name,
                   metrics=metrics, name=r'$\mathcal{L}_{SVGP}$', color="m",
                   its_label=its_label, its_cut=its_cut)
        methods.append(SVI)

    classic_methods = []
    if GP_str is not None and classic_metrics is not None:
        GP = \
            method(method_str=GP_str,
                   base=base,
                   runs=runs,
                   experiment=dataset_str,
                   metrics=classic_metrics, name='GP', color="green",
                   its_label=its_label, its_cut=its_cut)
        classic_methods.append(GP)

    if VFE_str is not None and classic_metrics is not None:
        VFE = \
            method(method_str=VFE_str,
                   base=base,
                   runs=runs,
                   experiment=dataset_str,
                   metrics=classic_metrics, name=r'$\mathcal{L}_{VFE}$',
                   color="darkorange",
                   its_label=its_label, its_cut=its_cut)
        classic_methods.append(VFE)

    return methods, classic_methods


def all_stats(names, methods, runs):
    metrics = [LL, RMSE, TIME]
    markers = [mark_max, mark_min, mark_min]
    for j, metric in enumerate(metrics):

        print(metric)

        for i, dataset in enumerate(names):
            dataset_stats(names[i], methods[i], runs,
                          metric=metric, marker=markers[j])

        print("")


def mark_max(methods, metric, runs, last_n_its):
    values = []
    max = None
    max_index = None
    for i, method in enumerate(methods):
        a = method[get_columns(metric, runs)].tail(last_n_its[metric]).values
        mean = a.mean()
        std = a.std()
        values.append((method.simple_name, mean, std))
        if max is None or mean > max:
            max = mean
            max_index = i

    return values, max_index


def mark_min(methods, metric, runs, last_n_its):
    values = []
    min = None
    min_index = None
    for i, method in enumerate(methods):
        a = method[get_columns(metric, runs)].tail(last_n_its[metric]).values
        mean = a.mean()
        std = a.std()
        values.append((method.simple_name, mean, std))
        if min is None or mean < min:
            min = mean
            min_index = i

    return values, min_index


def dataset_stats(name, methods, runs, metric, marker):
    last_n_its = {
        LL: 10,
        RMSE: 10,
        TIME: 1
    }

    assert methods[0].simple_name == LIA_to_PP_simple_name
    assert methods[1].simple_name == LIA_simple_name
    assert methods[2].simple_name == SVGP_simple_name

    values, highlighted_index = marker(methods, metric, runs, last_n_its)

    print("\\texttt{{ {0} }} & ".format(name), end=" ")
    for i in range(len(values)):
        name, mean, std = values[i]
        highligh_index = i == highlighted_index
        not_last_index = i < (len(methods) - 1)

        print("{0}$\pm${1} {2}".format(
            "\\textbf{{ {0:.4f} }}".format(
                mean) if highligh_index else "{0:.4f}".format(mean),
            "\\textbf{{ {0:.4f} }}".format(
                std) if highligh_index else "{0:.4f}".format(std),
            " & " if not_last_index else " \\\\ "), end=" ")

    if metric == TIME:
        print("& {0:.4f}".format(values[2][1] / values[1][1]), end=" ")

    print("\n\\hline")

def plot_syn_demo(dimensions,number_of_inducing_points,runs=1):

    name = "{0}D".format(dimensions)
    methods, _ = load(
        base="{0}/results".format(pathlib.Path().absolute()),
        # specify base folder
        dataset_str="gendata_{0}_100000_10000_0.5_1_0.01_-1_1".format(dimensions),
        # specify subfolder

        LIA_to_PP_str="LIA_to_PP_{0}_5000_0.001".format(number_of_inducing_points),
        # specify which methods should be loaded
        SVGP_str="SVI_{0}_5000_0.001_nat-grad_0.1".format(number_of_inducing_points),

        metrics=[TIME, RMSE, LL],  # which metrics should be loaded

        runs=1)

    fig = plt.figure(figsize=(12, 5))

    fontsize = 14
    cols = 1
    columns = 3
    k = 1

    ax = fig.add_subplot(cols, columns, k)
    k += 1
    ax.set_ylabel(name, fontsize=fontsize, rotation=0, labelpad=30)
    plot_methods(ax, methods, ITS, RMSE, title=RMSE_label,
                 fontsize=fontsize, runs=runs,xlabel=ITS_label)
    handles, labels = ax.get_legend_handles_labels()

    ax = fig.add_subplot(cols, columns, k)
    k += 1
    plot_methods(ax, methods, ITS, LL,xlabel=ITS_label,
                 fontsize=fontsize, runs=runs, title=LL_label)

    ax = fig.add_subplot(cols, columns, k)
    k += 1
    plot_methods(ax, methods, TIME, LL, title=LL_label,
                 fontsize=fontsize, runs=runs,xlabel=TIME_label)

    lgd = fig.legend(handles, labels, loc='lower center',
                     fontsize=fontsize,
                     bbox_to_anchor=(0, -0.1, 1, 1), ncol=3,
                     fancybox=True, shadow=True)

    fig.savefig('./results/{0}_synthetic_demo.png'.format(name),
                bbox_extra_artists=(lgd,),
                dpi=600, bbox_inches='tight')
    plt.close()

def plot_syn_extended_figure(methods_syn_5D,methods_syn_10D):

    fig = plt.figure(figsize=(10, 8))

    fontsize = 14
    cols = 2
    columns = 3
    k = 1

    ax = fig.add_subplot(cols, columns, k)
    k += 1
    ax.set_ylabel("5D", fontsize=fontsize, rotation=0, labelpad=30, )
    plot_methods(ax, methods_syn_5D, ITS, RMSE,title=RMSE_label,
                 fontsize=fontsize,runs=5,
               )
    handles, labels = ax.get_legend_handles_labels()

    ax = fig.add_subplot(cols, columns, k)
    k += 1
    plot_methods(ax, methods_syn_5D, ITS, LL,
                 fontsize=fontsize,runs=5,title=LL_label,
                )

    ax = fig.add_subplot(cols, columns, k)
    k += 1
    plot_methods(ax, methods_syn_5D, TIME, LL,title=LL_label,
                 fontsize=fontsize,runs=5,
              )

    ax = fig.add_subplot(cols, columns, k)
    k += 1
    ax.set_ylabel("10D", fontsize=fontsize, rotation=0, labelpad=30)
    plot_methods(ax, methods_syn_10D, ITS, RMSE,
                 fontsize=fontsize,runs=5,
                 xlabel=ITS_label)

    ax = fig.add_subplot(cols, columns, k)
    k += 1
    plot_methods(ax, methods_syn_10D, ITS, LL,
                 fontsize=fontsize,runs=5,
                 xlabel=ITS_label)

    ax = fig.add_subplot(cols, columns, k)
    k += 1
    plot_methods(ax, methods_syn_10D, TIME, LL,
                 fontsize=fontsize,runs=5,
                 xlabel=TIME_label)

    lgd = fig.legend(handles, labels, loc='lower center',
                     fontsize=fontsize,
                     bbox_to_anchor=(0, -0.02, 1, 1), ncol=3,
                     fancybox=True, shadow=True)

    # plt.show()
    fig.savefig('../plots/syn_extended2.png',
                bbox_extra_artists=(lgd,),
                dpi=600, bbox_inches='tight')
    plt.close()

