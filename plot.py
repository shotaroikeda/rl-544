import matplotlib.pyplot as plt
import os
import re
import numpy as np

BASE = os.path.join("report", "assets")

PLOT_DIRS = [
    "base",
    "lbfgs_hist",
    "lbfgs_alpha"
]

TITLES = [
    "Adam vs. LBFGS",
    "LBFGS with Different $h$",
    "LBFGS with Different $\\alpha$"
]

LEGENDS = [
    True,
    True,
    True
]

NPY_FILTER = re.compile('.*\\.npy', re.IGNORECASE)

ADAM_FILTER = re.compile('adam_([0-9\.]+)_([0-9\.]+)\.npy', re.IGNORECASE)
LBFGS_FILTER = re.compile('lbfgs_([0-9\.]+)_([0-9\.]+)_([0-9\.]+)\.npy', re.IGNORECASE)


def file_filter(s):
    match = ADAM_FILTER.match(s)
    if match is not None:
        grp = match.groups()
        grp = (float(grp[0]), float(grp[1]))
        return ('adam', ) + grp

    match = LBFGS_FILTER.match(s)
    if match is not None:
        grp = match.groups()
        grp = (float(grp[0]), float(grp[1]), int(grp[2]))
        return ('lbfgs',) + grp

    return None


def re_filter(regexp):
    def m(s):
        m = regexp.match(s)
        return m is not None

    return m


def main():
    for plt_dirs, title, use_legend in zip(PLOT_DIRS, TITLES, LEGENDS):
        exp_dir = os.path.join(BASE, plt_dirs)
        npy_files = filter(re_filter(NPY_FILTER), sorted(os.listdir(exp_dir)))

        plt.title(title)

        mixed_adam = False
        diff_alpha = False
        diff_lr = False
        diff_hist = False

        prev_alpha = None
        prev_lr = None
        prev_hist = None

        for npy_file in npy_files:
            params = file_filter(npy_file)
            if params[0] is 'adam':
                mixed_adam = True

            if params[0] is 'lbfgs':
                if prev_alpha is None:
                    prev_alpha, prev_lr, prev_hist = params[1:]
                else:
                    if not np.isclose(prev_alpha, params[1]):
                        diff_alpha = True
                    if not np.isclose(prev_lr, params[2]):
                        diff_lr = True
                    if not np.isclose(prev_hist, params[3]):
                        diff_hist = True

        for npy_file in npy_files:
            params = file_filter(npy_file)

            arr = np.load(os.path.join(BASE, plt_dirs, npy_file))
            mean_arr = arr.mean(0)
            var_arr = arr.std(0)

            label = ''

            if mixed_adam:
                label += '%s ' % (params[0])
            else:
                if diff_alpha:
                    label += ' $\\alpha={:.2f}$'.format(params[1])
                if diff_lr:
                    label += ' $\\eta={:.2f}$'.format(params[2])
                if diff_hist:
                    label += ' $h={:d}$'.format(params[3])

            x = np.arange(mean_arr.shape[0])
            plt.plot(x, mean_arr,
                     label=label)
            min_var = mean_arr - var_arr
            min_var[min_var < 0] = 0
            plt.fill_between(x, min_var, mean_arr + var_arr, alpha=0.2)

        plt.ylim((0, 250))
        plt.xlim((0, x[-1]))
        plt.legend()
        plt.savefig(os.path.join(BASE, plt_dirs) + '.png', dpi=150)
        plt.clf()


if __name__ == '__main__':
    main()
