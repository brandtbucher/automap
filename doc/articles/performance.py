import os
import sys
import timeit
import typing as tp
from itertools import repeat

import automap
from automap import AutoMap
from automap import FrozenAutoMap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())


class MapProcessor:
    NAME = ""
    SORT = -1

    def __init__(self, array: np.ndarray):
        self.array = array
        self.list = array.tolist()
        self.faml = FrozenAutoMap(self.list)
        self.fama = FrozenAutoMap(self.array)
        self.d = dict(zip(self.list, range(len(self.list))))


# -------------------------------------------------------------------------------
class FAMLInstantiate(MapProcessor):
    NAME = "FAM(L): instantiate"
    SORT = 0

    def __call__(self):
        fam = FrozenAutoMap(self.list)
        assert len(fam) == len(self.list)


class AMAInstantiate(MapProcessor):
    NAME = "AM(A): instantiate"
    SORT = 0

    def __call__(self):
        fam = AutoMap(self.array)
        assert len(fam) == len(self.list)


class FAMAInstantiate(MapProcessor):
    NAME = "FAM(A): instantiate"
    SORT = 0

    def __call__(self):
        fam = FrozenAutoMap(self.array)
        assert len(fam) == len(self.list)


class FAMAtolistInstantiate(MapProcessor):
    NAME = "FAM(Atolist): instantiate"
    SORT = 0

    def __call__(self):
        fam = FrozenAutoMap(self.array.tolist())
        assert len(fam) == len(self.list)


class DictInstantiate(MapProcessor):
    NAME = "Dict: instantiate"
    SORT = 0

    def __call__(self):
        d = dict(zip(self.list, range(len(self.list))))
        assert len(d) == len(self.list)


# -------------------------------------------------------------------------------
class FAMLLookup(MapProcessor):
    NAME = "FAM(L): lookup"
    SORT = 0

    def __call__(self):
        m = self.faml
        for k in self.list:
            _ = m[k]


class FAMALookup(MapProcessor):
    NAME = "FAM(A): lookup"
    SORT = 0

    def __call__(self):
        m = self.fama
        for k in self.list:
            _ = m[k]


class DictLookup(MapProcessor):
    NAME = "Dict: lookup"
    SORT = 0

    def __call__(self):
        m = self.d
        for k in self.list:
            _ = m[k]


# -------------------------------------------------------------------------------
class FAMLKeys(MapProcessor):
    NAME = "FAM(L): keys"
    SORT = 0

    def __call__(self):
        for v in self.faml.keys():
            pass


class FAMAKeys(MapProcessor):
    NAME = "FAM(A): keys"
    SORT = 0

    def __call__(self):
        for v in self.fama.keys():
            pass


class DictKeys(MapProcessor):
    NAME = "Dict: keys"
    SORT = 0

    def __call__(self):
        for v in self.d.keys():
            pass


# -------------------------------------------------------------------------------
class FAMLItems(MapProcessor):
    NAME = "FAM(L): items"
    SORT = 0

    def __call__(self):
        for k, v in self.faml.items():
            pass


class FAMAItems(MapProcessor):
    NAME = "FAM(A): items"
    SORT = 0

    def __call__(self):
        for k, v in self.fama.items():
            pass


class DictItems(MapProcessor):
    NAME = "Dict: items"
    SORT = 0

    def __call__(self):
        for k, v in self.d.items():
            pass


# -------------------------------------------------------------------------------
NUMBER = 100

from itertools import product


def seconds_to_display(seconds: float) -> str:
    seconds /= NUMBER
    if seconds < 1e-4:
        return f"{seconds * 1e6: .1f} (µs)"
    if seconds < 1e-1:
        return f"{seconds * 1e3: .1f} (ms)"
    return f"{seconds: .1f} (s)"


GROUPS = 4


def plot_performance(frame):
    fixture_total = len(frame["fixture"].unique())
    cat_total = len(frame["size"].unique())
    processor_total = len(frame["cls_processor"].unique())
    fig, axes = plt.subplots(cat_total, fixture_total)

    # cmap = plt.get_cmap('terrain')

    cmap = plt.get_cmap("plasma")
    color = cmap(np.arange(processor_total) / processor_total)
    # color = []
    # for i in range(GROUPS):
    #     for j in range(0, processor_total, GROUPS):
    #         k = i + j
    #         if k < len(color_raw):
    #             color.append(color_raw[i + j])

    # category is the size of the array
    for cat_count, (cat_label, cat) in enumerate(frame.groupby("size")):
        for fixture_count, (fixture_label, fixture) in enumerate(
            cat.groupby("fixture")
        ):
            ax = axes[cat_count][fixture_count]

            # set order
            fixture["sort"] = [f.SORT for f in fixture["cls_processor"]]
            fixture = fixture.sort_values("sort")

            results = fixture["time"].values.tolist()
            names = [cls.NAME for cls in fixture["cls_processor"]]
            # x = np.arange(len(results))
            names_display = names
            post = ax.bar(names_display, results, color=color)

            # density, position = fixture_label.split('-')
            # cat_label is the size of the array
            title = f"{cat_label:.0e}\n{fixture_label}"

            ax.set_title(title, fontsize=6)
            ax.set_box_aspect(0.6)  # makes taller tan wide
            time_max = fixture["time"].max()
            ax.set_yticks([0, time_max * 0.5, time_max])
            ax.set_yticklabels(
                [
                    "",
                    seconds_to_display(time_max * 0.5),
                    seconds_to_display(time_max),
                ],
                fontsize=6,
            )
            # ax.set_xticks(x, names_display, rotation='vertical')
            ax.tick_params(
                axis="x",
                which="both",
                bottom=False,
                top=False,
                labelbottom=False,
            )

    fig.set_size_inches(8, 4)  # width, height
    fig.legend(post, names_display, loc="center right", fontsize=6)
    # horizontal, vertical
    fig.text(0.05, 0.96, f"AutoMap Performance: {NUMBER} Iterations", fontsize=10)
    fig.text(0.05, 0.90, get_versions(), fontsize=6)

    fp = "/tmp/automap.png"
    plt.subplots_adjust(
        left=0.075,
        bottom=0.05,
        right=0.80,
        top=0.80,
        wspace=0.7,  # width
        hspace=0.4,
    )
    # plt.rcParams.update({'font.size': 22})
    plt.savefig(fp, dpi=300)

    if sys.platform.startswith("linux"):
        os.system(f"eog {fp}&")
    else:
        os.system(f"open {fp}")


# -------------------------------------------------------------------------------


class FixtureFactory:
    NAME = ""

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        raise NotImplementedError()

    @classmethod
    def get_label_array(cls, size: int) -> tp.Tuple[str, np.ndarray]:
        array = cls.get_array(size)
        return cls.NAME, array


class FFInt64(FixtureFactory):
    NAME = "int64"

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(size, dtype=np.int64)
        array.flags.writeable = False
        return array


class FFInt32(FixtureFactory):
    NAME = "int32"

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.arange(size, dtype=np.int32)
        array.flags.writeable = False
        return array


class FFFloat64(FixtureFactory):
    NAME = "float64"

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = (np.arange(size) * 0.5).astype(np.float64)
        array.flags.writeable = False
        return array


class FFFloat32(FixtureFactory):
    NAME = "float32"

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = (np.arange(size) * 0.5).astype(np.float32)
        array.flags.writeable = False
        return array


class FFString(FixtureFactory):
    NAME = "string"

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        array = np.array([hex(e) for e in range(size)])
        array.flags.writeable = False
        return array


class FFObject(FixtureFactory):
    NAME = "object"

    @staticmethod
    def get_array(size: int) -> np.ndarray:
        ints = np.arange(size)
        array = ints.astype(object)

        target = 1 == ints % 3
        array[target] = ints[target] * 0.5

        target = 2 == ints % 3
        array[target] = np.array([hex(e) for e in ints[target]])

        array.flags.writeable = False
        return array


def get_versions() -> str:
    import platform

    return f"OS: {platform.system()} / AutoMap / NumPy: {np.__version__}\n"


CLS_PROCESSOR = (
    FAMLInstantiate,
    # AMAInstantiate,
    FAMAInstantiate,
    # FAMAtolistInstantiate,
    DictInstantiate,
    FAMLLookup,
    FAMALookup,
    DictLookup,
    FAMLKeys,
    FAMAKeys,
    DictKeys,
)

CLS_FF = (
    FFInt32,
    FFInt64,
    FFFloat64,
    FFString,
    FFObject,
)


def run_test():
    records = []
    for size in (100, 10_000, 1_000_000):
        for ff in CLS_FF:
            fixture_label, fixture = ff.get_label_array(size)
            for cls in CLS_PROCESSOR:
                runner = cls(fixture)

                record = [cls, NUMBER, fixture_label, size]
                print(record)
                try:
                    result = timeit.timeit(f"runner()", globals=locals(), number=NUMBER)
                except OSError:
                    result = np.nan
                finally:
                    pass
                record.append(result)
                records.append(record)

    f = pd.DataFrame.from_records(
        records, columns=("cls_processor", "number", "fixture", "size", "time")
    )
    print(f)
    plot_performance(f)


if __name__ == "__main__":

    run_test()
