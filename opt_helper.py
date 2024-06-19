# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import os
from operator import mul, sub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn import linear_model

default = {
    "_RICE_CONSTANT": {
        "xgamma": 0.3,  # in CAP Eq 5 the capital elasticty
        # A rice data
        "xA_0": 0,
        "xg_A": 0,
        "xdelta_A": 0.0214976314392836,
        # L
        "xL_0": 1397.715000,  # in POP population at the staring point
        "xL_a": 1297.666000,  # in POP the expected population at convergence
        "xl_g": 0.04047275402855734,  # in POP control the rate to converge
        # K
        "xK_0": 93.338152,
        "xa_1": 0,
        "xa_2": 0.00236,
        "xa_3": 2,
        # xsigma_0: 0.5201338309755572
        "xsigma_0": 0.215,
    }
}


def logdiff(List):
    loglist = np.log(List)
    return np.array([loglist[i + 1] - loglist[i] for i in range(len(List) - 1)])


def get_pop_lg(
    data, Las, unit=1, stop_condition=0.0001, trim_start=0, trim_end=0, verbose=0
):
    """
    @param country: str, the country code, e.g. "USA"
    @param Las: dict, record the converged population, e.g. {"USA": 433850000}, unit: 1
    @param unit: the calculation unit, e.g. 1000000 means milion
    @param stop_condition: float or int, if int, it's the total step for the simulation;
            if float, it's the tolerance as convergence condition for the simulation
    @param verbose, can be 0, 1, or larger. verbose=0: only return the para,
            =1: return the param and insample MAPE and plot them out,
            =2: return the param and insample MAPE and run a simulation with stop_condition and plot them out
    """

    def pop(ini, lg, la, stop_condition=0.0001):
        pops = [ini]
        if isinstance(stop_condition, int):
            max_step = stop_condition
            for i in range(max_step - 1):
                pops.append(pops[-1] * ((1 + la) / (1 + pops[-1])) ** lg)
        elif isinstance(stop_condition, float):
            tol = stop_condition
            while True:
                pops.append(pops[-1] * ((1 + la) / (1 + pops[-1])) ** lg)
                if pops[-1] / pops[-2] - 1 < tol:
                    break
        return pops

    La = Las / unit
    dataList = np.array(data)
    if trim_end == 0:
        dataList = dataList[trim_start:]
    else:
        dataList = dataList[trim_start:trim_end]
    Y = logdiff(dataList)
    X = np.log(1 + La) - np.log(1 + dataList[:-1]).reshape(-1, 1)
    reg = linear_model.LinearRegression(fit_intercept=False)
    reg.fit(X, Y)
    if verbose:
        plt.plot(dataList, color="r", label="real data")
        insample_est = []
        tmp = dataList[0]
        for i in range(len(dataList)):
            insample_est.append(tmp)
            tmp = tmp * ((1 + La) / (1 + tmp)) ** reg.coef_[0]
        plt.plot(insample_est, color="b", label="est data")
        if verbose > 1:
            until_converge = pop(
                dataList[0], reg.coef_[0], La, stop_condition=stop_condition
            )
            plt.plot(until_converge, color="g", label="est data")
            return (
                reg.coef_[0],
                MAPE(dataList, insample_est),
                insample_est,
                until_converge,
            )
        else:
            return reg.coef_[0], MAPE(dataList, insample_est)
    return reg.coef_[0]


def get_gA_deltaA(
    data,
    delta=1,
    stop_condition=0.0001,
    rescale_unit=13.26,
    delta_A_lower=0.005,
    verbose=False,
):
    def tfp(ini, params, delta=1, stop_condition=0.0001):
        g_A, delta_A = params
        tfps = [ini]
        if isinstance(stop_condition, int):
            max_step = stop_condition
            for i in range(max_step - 1):
                tfps.append(
                    tfps[-1]
                    * (
                        np.exp(0.0033)
                        + g_A
                        * np.exp(-(delta_A**2 + delta_A_lower) * delta * len(tfps))
                    )
                )
        elif isinstance(stop_condition, float):
            tol = stop_condition
            while True:
                tfps.append(
                    tfps[-1]
                    * (
                        np.exp(0.0033)
                        + g_A
                        * np.exp(-(delta_A**2 + delta_A_lower) * delta * len(tfps))
                    )
                )
                if tfps[-1] / tfps[-2] - 1 < tol:
                    break
        return tfps

    data = np.array(data)

    def target(x):
        return sum(
            np.square(
                data[1:] / data[:-1]
                - np.exp(0.0033)
                - x[0]
                * np.exp(
                    -(x[1] ** 2 + delta_A_lower)
                    * delta
                    * np.array(range(len(data) - 1))
                )
            )
        )

    x0 = np.array([0.01, 0.01])
    res = minimize(
        target, x0, method="nelder-mead", options={"xatol": 1e-8, "disp": True}
    )
    if verbose:
        plt.plot(data, color="r", label="real data")
        insample_est = []
        tmp = data[0]
        for i in range(len(data)):
            insample_est.append(tmp)
            tmp = tmp * (
                np.exp(0.0033)
                + res.x[0] * np.exp(-(res.x[1] ** 2 + delta_A_lower) * delta * (i))
            )
        plt.plot(insample_est, color="b", label="est data")
        if verbose > 1:
            until_converge = tfp(data[0], res.x, delta=1, stop_condition=stop_condition)
            plt.plot(until_converge, color="g", label="est data")
            return (
                [res.x[0], res.x[1] ** 2 + delta_A_lower],
                res.fun,
                MAPE(data, insample_est),
                insample_est,
                until_converge,
            )
        else:
            return [res.x[0], res.x[1] ** 2 + delta_A_lower], res.fun
    else:
        return [res.x[0], res.x[1] ** 2 + delta_A_lower]


def write_yaml_files(pos_s, save_path, default_dict=default, ext=".yml"):
    import os

    import yaml

    result = default_dict.copy()
    for k, v in pos_s.items():
        result["_RICE_CONSTANT"].update(v)
        for x, y in result["_RICE_CONSTANT"].items():
            if x in ["xa_1", "xa_3"]:
                continue
            result["_RICE_CONSTANT"][x] = float(y)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_path = os.path.join(save_path, str(k) + ext)
        with open(file_path, "w") as file:
            yaml.dump(result, file)


def merge_region(datadict, gamma=0.3, mode="classic"):
    """
    Ai_s, 2d-nparray, first dim is region, second dim is time, tfp
    Ki_s, 2d-nparray, first dim is region, second dim is time, capital
    Li_s, 2d-nparray, first dim is region, second dim is time, labor
    La_s, 1d-nparray, the dim represents regions
    sigmai_s, 1d-nparray, the dim represents regions
    """
    Ai_s = datadict["TS_A"]
    Ki_s = datadict["TS_K"]
    Li_s = datadict["TS_L"]
    La_s = datadict["La_s"]
    sigmai_s = datadict["sigmai_s"]
    tax_s = datadict["tax_s"]
    mitigation_s = datadict["mitigation_s"]
    saving_s = datadict["saving_s"]
    assert mode in ["classic", "efficient"], "only support classic or efficient mode!"
    assert (
        len(Ai_s) == len(Ki_s) == len(Li_s) == len(La_s) == len(sigmai_s)
    ), "These lists much have the same length!"
    Ai_s, Ki_s, Li_s, La_s, sigmai_s = (
        np.array(Ai_s),
        np.array(Ki_s),
        np.array(Li_s),
        np.array(La_s),
        np.array(sigmai_s),
    )
    Yi_s = Ai_s * Ki_s**gamma * Li_s ** (1 - gamma)
    N, T = Ai_s.shape
    if mode == "classic":
        Ks = np.sum(Ki_s, axis=0)
        Ls = np.sum(Li_s, axis=0)
        Las = np.sum(La_s)
        Ys = np.sum(Yi_s, axis=0)
        As = Ys / (Ks**gamma * Ls ** (1 - gamma))
        sigmas = np.sum(Yi_s[:, -1] * sigmai_s) / Ys[-1]
        taxs = calculate_geometric_mean(tax_s)
        mitigations = calculate_geometric_mean(mitigation_s)
        savings = calculate_geometric_mean(saving_s)
    elif mode == "efficient":
        AKs = np.sum(Ai_s * Ki_s, axis=0)
        ALs = np.sum(Ai_s * Li_s, axis=0)
        Ai_s_sort = np.sort(Ai_s, axis=0)[::-1]
        if N <= 3:
            As = Ai_s_sort[0, :]
        elif N <= 5:
            As = np.mean(Ai_s_sort[0:2, :], axis=0)
        else:
            As = np.mean(Ai_s_sort[0:3, :], axis=0)
        Ks = AKs / As
        Ls = ALs / As
        Las = np.sum(La_s) * Ls[-1] / sum(Li_s, axis=0)[-1]
        sigmas = (
            np.sum(
                sigmai_s
                * Ai_s[:, -1]
                * Ki_s[:, -1] ** gamma
                * Li_s[:, -1] ** (1 - gamma)
            )
            / AKs[-1] ** gamma
            * ALs[-1] ** (1 - gamma)
        )
        taxs = calculate_geometric_mean(tax_s)
        mitigations = calculate_geometric_mean(mitigation_s)
        savings = calculate_geometric_mean(saving_s)
    # return As, Ks, Ls, Las, sigmas, taxs, mitigations, savings
    return {
        "As": As,
        "Ks": Ks,
        "Ls": Ls,
        "Las": Las,
        "sigmas": sigmas,
        "taxs": taxs,
        "mitigations": mitigations,
        "savings": savings,
    }


def packup_regions(raw_data, groupnum, countryclass, exc_code, raw_results):
    output = {}
    starts = []
    ends = []
    for token in ["TS_A", "TS_K", "TS_L"]:
        for c in countryclass[groupnum]:
            if c in exc_code:
                continue
            starts.append(raw_data[c][token][3])
            ends.append(raw_data[c][token][4])
    s = max(starts)
    e = min(ends)

    for token in ["TS_A", "TS_K", "TS_L"]:
        output[token] = []
        for c in countryclass[groupnum]:
            if c in exc_code:
                continue
            for i in range(s, e + 1):
                if i == s:
                    output[token].append([])
                output[token][-1].append(raw_data[c][token][1]["YR" + str(i)])
    output["La_s"] = [
        raw_results[c]["La"] for c in countryclass[groupnum] if c not in exc_code
    ]
    output["sigmai_s"] = [
        raw_results[c]["TS_sigma"][0][-1]
        for c in countryclass[groupnum]
        if c not in exc_code
    ]
    output["tax_s"] = [
        raw_results[c]["tax"] for c in countryclass[groupnum] if c not in exc_code
    ]
    output["mitigation_s"] = [
        raw_results[c]["mitigation"]
        for c in countryclass[groupnum]
        if c not in exc_code
    ]
    output["saving_s"] = [
        raw_results[c]["saving"] for c in countryclass[groupnum] if c not in exc_code
    ]
    return output


def merge_region_dict(datadict, gamma=0.3, mode="classic"):
    return merge_region(
        datadict,
        gamma=gamma,
        mode=mode,
    )


def calculate_geometric_mean(rates):
    """
    Calculate the geometric mean of a list or ndarray of rates.

    Parameters:
    rates (list or ndarray): A list or ndarray of rates.

    Returns:
    float: The geometric mean of the rates.
    """
    # Convert the input to a numpy array to ensure compatibility with scipy's gmean function
    rates_array = np.array(rates, dtype=float)

    # Calculate the geometric mean
    return (1 + rates_array).prod() ** (1.0 / len(rates_array)) - 1


def split_region(datadict, code, start, end, splits=[], gamma=0.3):
    A, K, L = [], [], []
    for i in range(start, end + 1):
        A.append(datadict[code]["TS_A"][1]["YR" + str(i)])
        K.append(datadict[code]["TS_K"][1]["YR" + str(i)])
        L.append(datadict[code]["TS_L"][1]["YR" + str(i)])
    A, K, L = np.array(A), np.array(K), np.array(L)
    La = datadict[code]["La"]
    Y = A * K**gamma * L ** (1 - gamma)
    splits = np.array(splits)
    if len(splits) == 0:
        splits = np.random.rand(4)
    if sum(splits) != 1:
        splits = splits / sum(splits)
    LEN = len(splits)
    Ys = Y.reshape(1, -1)
    Ys = np.repeat(Ys, LEN, axis=0)
    for i in range(len(splits)):
        Ys[i] = Ys[i] * splits[i]
    Ls = L.reshape(1, -1)
    Ls = np.repeat(Ls, LEN, axis=0)
    for i in range(len(splits)):
        Ls[i] = Ls[i] * splits[i]
    multiplier = np.clip(np.exp(np.random.normal(0.5, 0.5, LEN)), 0.75, 2)
    As = A.reshape(1, -1)
    As = np.repeat(As, LEN, axis=0)
    for i in range(len(multiplier)):
        As[i] = As[i] * multiplier[i]
    Ks = (Ys / (As * Ls ** (1 - gamma))) ** (1 / gamma)
    Las = La * splits
    return As, Ks, Ls, Las


def save(obj, filename):
    import pickle

    try:
        from deepdiff import DeepDiff
    except:
        os.system("pip install deepdiff")
        from deepdiff import DeepDiff
    with open(filename, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)  # highest protocol
    with open(filename, "rb") as file:
        z = pickle.load(file)
    assert (
        DeepDiff(obj, z, ignore_string_case=True) == {}
    ), "there is something wrong with the saving process"
    return


def load(filename):
    # import pickle
    import pickle5 as pickle  # on Python 3.7

    with open(filename, "rb") as file:
        z = pickle.load(file)
    return z


def get_mean_std(list_of_dict):
    k = list_of_dict[0].keys()
    v = [list(x.values()) for x in list_of_dict]
    mean = [sum(x) / len(v) for x in zip(*v)]
    demean = [list(map(sub, v[i], mean)) for i in range(len(v))]
    muls = [list(map(mul, demean[i], demean[i])) for i in range(len(v))]
    sqr = [sum(x) / len(v) for x in zip(*muls)]
    sqrt = [np.sqrt(x) for x in sqr]
    mean_dict = dict(zip(k, mean))
    std_dict = dict(zip(k, sqrt))
    return mean_dict, std_dict


def get_upper_lower_bounds(list_of_dict, n=1.96):
    mean_dict, std_dict = get_mean_std(list_of_dict)
    upper = {k: mean_dict[k] + n * std_dict[k] for k in mean_dict.keys()}
    lower = {k: mean_dict[k] - n * std_dict[k] for k in mean_dict.keys()}
    return upper, lower, mean_dict


def plot_fig_with_bounds(
    variable,
    y_label,
    list_of_dict_off=None,
    list_of_dict_on=None,
    title=None,
    idx=0,
    x_label="year",
    skips=3,
    line_colors=["#0868ac", "#7e0018"],
    region_colors=["#7bccc4", "#ffac3b"],
    start=2020,
    alpha=0.5,
    is_grid=True,
    is_save=True,
    delta=5,
):
    ax = plt.axes()
    ax.spines["bottom"].set_color("#676767")  # dark grey
    ax.spines["top"].set_color("#676767")
    ax.spines["right"].set_color("#676767")
    ax.spines["left"].set_color("#676767")
    year = np.array(range(len(list(list_of_dict_off[0].values())[0]))) * delta + start
    if list_of_dict_off is not None:
        upper_off, lower_off, mean_off = get_upper_lower_bounds(list_of_dict_off)
        if idx == -1:
            plt.plot(
                year,
                mean_off[variable][...],
                label="no negotiation",
                linestyle="--",
                color=line_colors[0],
            )
            plt.fill_between(
                year,
                lower_off[variable][...],
                upper_off[variable][...],
                color=region_colors[0],
                alpha=0.5,
            )
        else:
            plt.plot(
                year,
                mean_off[variable][..., idx],
                label="no negotiation",
                linestyle="--",
                color=line_colors[0],
            )
            plt.fill_between(
                year,
                lower_off[variable][..., idx],
                upper_off[variable][..., idx],
                color=region_colors[0],
                alpha=0.5,
            )
    if list_of_dict_on is not None:
        upper_on, lower_on, mean_on = get_upper_lower_bounds(list_of_dict_on)
        if idx == -1:
            plt.plot(
                year,
                mean_on[variable][...][::skips],
                label="with negotiation",
                color=line_colors[1],
            )
            plt.fill_between(
                year,
                lower_on[variable][...][::skips],
                upper_on[variable][...][::skips],
                color=region_colors[1],
                alpha=0.5,
            )
        else:
            plt.plot(
                year,
                mean_on[variable][..., idx][::skips],
                label="with negotiation",
                color=line_colors[1],
            )
            plt.fill_between(
                year,
                lower_on[variable][..., idx][::skips],
                upper_on[variable][..., idx][::skips],
                color=region_colors[1],
                alpha=0.5,
            )

    plt.legend(loc=2)
    if is_grid:
        plt.grid(color="#d3d3d3")  # light grey
    plt.ylabel(y_label)
    plt.xlabel("Year")
    if title is not None:
        plt.title(title)
    if is_save:
        plt.savefig("{}.pdf".format(title))
    return


def plot_result(variable, nego_off=None, nego_on=None, k=0):
    """
    variable can be a list of string or a single string.

    When it is a string. It should be one of the list(nego_off.keys()) that one wants to show
    nego_off and nego_on are dictionaries from the training results, in details
        nego_off = trainer_off.fetch_episode_states(desired_outputs)
        nego_on = trainer_on.fetch_episode_states(desired_outputs)
    k is the index of the vectored variable that one wants to show, please keep it as 0 if
    variable not in ["global_temperature", "global_carbon_mass"]

    When it is a list of string. It should be a subset of list(nego_off.keys()) which includes
    variables that one wants to show. It is equalvalent to iterate through the list of string and draw
    each variable one by one.
    """
    if isinstance(variable, list):
        for i in range(len(variable)):
            try:
                plot_result(variable[i], nego_off=nego_off, nego_on=nego_on, k=k)
            except:
                print("Error:", variable[i])
    else:
        if variable not in ["global_temperature", "global_carbon_mass"]:
            assert k == 0, f"There are only 1 variable record for {variable}"
        if variable == "global_temperature":
            assert k <= 1, f"There are only 2 variable records for global_temperature"
        if variable == "global_carbon_mass":
            assert k <= 2, f"There are only 3 variable records for global_carbon_mass"

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        legends = []
        if nego_off is not None:
            ax.plot(nego_off[variable][..., k])
            legends.append("nego_off")
        if nego_on is not None:
            ax.plot(nego_on[variable][..., k][::3])
            legends.append("nego_on")
        ax.legend(legends)
        ax.grid()
        ax.set_title(f"{variable}".replace("_", " ").title())
        ax.set_xlabel("Steps")
        ax.set_ylabel(variable)
        return fig, ax


def plot_training_curve(
    data, mertic, submission_file_name, start=None, end=None, return_data=False
):
    """
    plotting mertics collected in a dictionary from the training procedure. Below are some of the available metrics:
    mertics = ['Iterations Completed',
     'VF loss coefficient',
     'Entropy coefficient',
     'Total loss',
     'Policy loss',
     'Value function loss',
     'Mean rewards',
     'Max. rewards',
     'Min. rewards',
     'Mean value function',
     'Mean advantages',
     'Mean (norm.) advantages',
     'Mean (discounted) returns',
     'Mean normalized returns',
     'Mean entropy',
     'Variance explained by the value function',
     'Gradient norm',
     'Learning rate',
     'Mean episodic reward',
     'Mean policy eval time per iter (ms)',
     'Mean action sample time per iter (ms)',
     'Mean env. step time per iter (ms)',
     'Mean training time per iter (ms)',
     'Mean total time per iter (ms)',
     'Mean steps per sec (policy eval)',
     'Mean steps per sec (action sample)',
     'Mean steps per sec (env. step)',
     'Mean steps per sec (training time)',
     'Mean steps per sec (total)'
     ]
    """
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    if data is None:
        data = get_training_curve(submission_file_name)

    if start is None:
        start = 0

    if end is None:
        ax.plot(data["Iterations Completed"][start:], data[mertic][start:])
    else:
        ax.plot(data["Iterations Completed"][start:end], data[mertic][start:end])

    ax.grid()
    ax.set_xlabel("iteration")
    ax.set_ylabel(mertic)

    if return_data:
        return fig, ax, data
    else:
        return fig, ax


def get_training_curve(submission_file_name):
    """
    get the metrics collected in a dictionary from the training procedure from the zip submission file.
    """
    import json
    import shutil
    import zipfile

    if "zip" != submission_file_name.split(".")[-1]:
        submission_file_name = submission_file_name + ".zip"
    # path_ = os.path.join("./Submissions/", submission_file_name)
    path_ = submission_file_name
    assert os.path.exists(
        path_
    ), f"This files is not available. Please check the path: {path_}."
    with zipfile.ZipFile(path_, "r") as zip_ref:
        # unzip_path = os.path.join(
        #     "./Submissions/", os.path.basename(path_).split(".")[0]
        # )
        unzip_path = path_[:-4]
        if not os.path.exists(unzip_path):
            os.makedirs(unzip_path)
        zip_ref.extractall(unzip_path)

    json_path = os.path.join(unzip_path, "results.json")
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = [json.loads(line) for line in f]
    shutil.rmtree(unzip_path)
    l = len(json_data)
    data = {}
    for k in json_data[0].keys():
        if isinstance(json_data[0][k], (int, float)):
            data[k] = [json_data[i][k] for i in range(l)]
        elif isinstance(json_data[0][k], dict):
            for k_ in json_data[0][k].keys():
                data[k_] = [json_data[i][k][k_] for i in range(l)]
    return data


def make_grid_plot(
    matrix_time_by_feature,
    feature_label="",
    xlabel="Year",
    ylabel="Value",
    cols=4,
    fig_scale=4,
):
    """
    Creates a matplotlib grid plot that plots each time series for each region.
    """
    timesteps, n_features = matrix_time_by_feature.shape

    rows = n_features // cols + 1
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(fig_scale * cols, fig_scale * rows),
        squeeze=False,
        sharey=True,
    )
    idx = 0
    print(f"Plotting for {n_features} features")
    for col in range(cols):
        if idx >= n_features:
            break
        for row in range(rows):
            ax = axes[row, col]
            ax.plot(matrix_time_by_feature[:, idx])
            ax.set_title(f"{feature_label} {idx}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid()

            idx += 1
            if idx >= n_features:
                break

    fig.tight_layout()
    return fig


def make_aggregate_data_across_three_groups(
    data_ts, group_1, group_2, group_3, n_steps=61, n_features=27
):
    n_groups = 3  # feature indices as defined in group_1, group_2, group_3
    aggregate_ts = dict()

    for key, value in data_ts.items():

        if value.shape == (n_steps, n_features):

            lo_data = value[:, group_1]
            med_data = value[:, group_2]
            hi_data = value[:, group_3]

            _aggregate_data = np.zeros((n_steps, n_groups))

            _aggregate_data[:, 0] = np.mean(lo_data, axis=1)
            _aggregate_data[:, 1] = np.mean(med_data, axis=1)
            _aggregate_data[:, 2] = np.mean(hi_data, axis=1)

            aggregate_ts[key] = _aggregate_data

    return aggregate_ts


def compute_correlation_across_groups(
    aggregate_stats_across_groups,
    data_ts,
    feature_name,
    do_plot=False,
):
    all_x = []
    all_y = []

    for group_idx, group in enumerate(groups):

        var_x = aggregate_stats_across_groups[::3, group_idx].mean()
        var_y = (
            data_ts[feature_name][::3, group]
            .sum(axis=0, keepdims=True)
            .mean(axis=1, keepdims=True)
        )

        all_x.append(var_x)
        all_y.append(var_y[0, 0])

    if do_plot:
        plt.scatter(all_x, all_y)

    """Give the correlation r2 between var_x and var_y"""
    r2 = np.corrcoef(all_x, all_y)[0, 1] ** 2

    return r2


# input a datasource, countrycode, token and output the coressponding datalist
def get_data_list(datasource, code, token, start=None, end=None):
    df = datasource
    if start is None:
        start = 1960
    if end is None:
        end = 2022
    years = ["YR" + str(i) for i in range(start, end)]
    translate = {
        "A": "ATFP",
        "K": "CM.MKT.LCAP.CD",
        "L": "SP.POP.TOTL",
        "Y": "NY.GDP.MKTP.CD",
        "sigma": "EN.ATM.CO2E.KD.CD",
        "C": "NE.CON.TOTL.ZS",
    }
    x = df[df["series"] == translate[token]][df["economy"] == code][years].values[0]
    dictx = {years[i]: x[i] for i in range(len(years))}
    clean_dict = {k: dictx[k] for k in dictx if not np.isnan(dictx[k])}
    # the following code is to clean up nan
    # e.g. we want to get from YR2003 to YR2018 because it is the most recent non-interuptted time series we can get
    #   {'YR1998': nan,
    #   'YR1999': nan,
    #   'YR2000': 4.7781155553647325,
    #   'YR2001': 4.9625160616028765,
    #   'YR2002': nan,
    #   'YR2003': 6.304385185367457,
    #   'YR2004': 6.969212836006577,
    #   'YR2005': 7.6477058929919774,
    #   'YR2006': 7.504520232482988,
    #   'YR2007': 8.317624876841197,
    #   'YR2008': 11.719807056127324,
    #   'YR2009': 6.912274757057348,
    #   'YR2010': 7.994212407175231,
    #   'YR2011': 9.736911695960446,
    #   'YR2012': 10.398433320757619,
    #   'YR2013': 12.063290518668204,
    #   'YR2014': 10.653260323582714,
    #   'YR2015': 11.37606978711656,
    #   'YR2016': 10.119059765144947,
    #   'YR2017': 10.686691489670057,
    #   'YR2018': 10.580970189286578,
    #   'YR2019': nan
    #   'YR2020': nan,
    #   'YR2021': nan}
    z = []
    j = -1
    flag = False
    for i in range(len(x) - 1, -1, -1):  # iterate from the rear
        if not pd.isnull(x[i]):
            flag = True  # we have the first normal number
            if j == -1:
                j = i
        if pd.isnull(x[i]) and not flag:
            continue
        if pd.isnull(x[i]) and flag:
            break  # break when we find the first nan after we find the first normal number
        else:
            z.append(x[i])

    out = list(reversed(z))
    return out, dictx, clean_dict, int(years[i + 1][2:]), int(years[j][2:])


def get_Y_C_L_pairs(df, codes, year=None, exc_code=[]):
    if year is None:
        year = "YR2020"
    else:
        year = "YR" + str(year)
    train_data = []
    train_label = []
    train_codes = []
    test_data = []
    test_codes = []
    codes_ = []
    for code in codes:
        if code in exc_code:
            continue
        else:
            label = get_data_list(df, code, "C")[1][year]
            if pd.isnull(label):
                test_data.append(
                    [
                        get_data_list(df, code, "Y")[1][year],
                        get_data_list(df, code, "L")[1][year],
                    ]
                )
                test_codes.append(code)
            else:
                train_data.append(
                    [
                        get_data_list(df, code, "Y")[1][year],
                        get_data_list(df, code, "L")[1][year],
                    ]
                )
                train_label.append(label)
                train_codes.append(code)
    return (
        np.array(train_data),
        np.array(train_label),
        np.array(test_data),
        dict(zip(train_codes, train_data)),
        dict(zip(test_codes, test_data)),
    )


def get_env_data_list(df, code):
    """
    Function to get the geo_avg_sum_indicator value for a given ISO3 code from the dataframe.

    :param df: Pandas DataFrame containing the data.
    :param code: String, the ISO3 code for which the geo_avg_sum_indicator is needed.
    :return: float, the geo_avg_sum_indicator value for the given ISO3 code.
    """
    # Filter the dataframe for the given code and retrieve the geo_avg_sum_indicator value
    value = df[df["ISO3"] == code]["geo_avg_sum_indicator"].values
    # If the value is found, return it, otherwise return None
    return float(value[0]) / 100 if len(value) > 0 else 0


def extract_group_memberships(csv_path):
    """
    Reads a CSV file and extracts group memberships of countries.

    Args:
    csv_path (str): Path to the CSV file.

    Returns:
    dict: A dictionary with group names as keys and lists of country ISO codes as values.
    """
    df = pd.read_csv(csv_path)

    # Initialize an empty dictionary to hold the group memberships
    group_memberships = {}

    # Iterate over the columns (excluding the first two columns which are 'Country' and 'ISO')
    for group in df.columns[2:]:
        # Filter out the unnamed columns that might not represent groups
        if "Unnamed" not in group:
            # Get the ISO codes of countries where the group value is 1
            members = df[df[group] == 1]["ISO"].tolist()
            group_memberships[group] = members

    return group_memberships


import json
import yaml

# Load the JSON file
with open("csv_asset/20_import_2016.json", "r") as json_file:
    import_data_dict = json.load(json_file)
with open("csv_asset/20_export_2016.json", "r") as json_file:
    export_data_dict = json.load(json_file)


# Function to update a single YAML file with new data
def update_yaml_file(file_path, data_im, data_ex):
    try:
        with open(file_path, "r") as yaml_file:
            current_data = yaml.safe_load(yaml_file) or {}
    except FileNotFoundError:
        current_data = {}

    current_data["_RICE_CONSTANT"]["ximport"] = data_im
    current_data["_RICE_CONSTANT"]["xexport"] = data_ex
    with open(file_path, "w") as yaml_file:
        yaml.dump(current_data, yaml_file, sort_keys=False)
    return True
