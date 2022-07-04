# Copyright (c) 2022, salesforce.com, inc and MILA.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause


import numpy as np
from sklearn import linear_model
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

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


def merge_region(Ai_s, Ki_s, Li_s, La_s, sigmai_s, gamma=0.3, mode="classic"):
    """
    Ai_s, 2d-nparray, first dim is region, second dim is time, tfp
    Ki_s, 2d-nparray, first dim is region, second dim is time, capital
    Li_s, 2d-nparray, first dim is region, second dim is time, labor
    La_s, 1d-nparray, the dim represents regions
    sigmai_s, 1d-nparray, the dim represents regions
    """
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
    return As, Ks, Ls, Las, sigmas


def merge_region_dict(datadict, gamma=0.3, mode="classic"):
    return merge_region(
        datadict["TS_A"],
        datadict["TS_K"],
        datadict["TS_L"],
        datadict["La_s"],
        datadict["sigmai_s"],
        gamma=0.3,
        mode="classic",
    )


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
    import pickle

    with open(filename, "rb") as file:
        z = pickle.load(file)
    return z


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

        plt.figure()
        legends = []
        if nego_off is not None:
            plt.plot(nego_off[variable][..., k])
            legends.append("nego_off")
        if nego_on is not None:
            plt.plot(nego_on[variable][..., k][::3])
            legends.append("nego_on")
        plt.legend(legends)
        plt.grid()
        plt.ylabel(variable)
