# PSL
import os
import glob
import time
import warnings
import traceback

from multiprocessing import Pool

# Others
import netCDF4


def make_plot(radar):
    dtime = netCDF4.num2date(radar.time['data'][0], radar.time['units'])

    filename = f"dealias_{dtime.strftime('%Y%m%d_%H%M')}.png"
    outfile = os.path.join(OUTPATH, filename)
    if os.path.exists(outfile):
        print(f"{outfile} already exists.")
        return None

    gr = pyart.graph.RadarDisplay(radar)
    fig, ax = pl.subplots(7, 2, figsize=(12, 35), sharey=True, sharex=True)
    ax = ax.flatten()

    for mysweep in range(7):
        gr.plot_ppi("VEL", ax=ax[2 * mysweep], cmap="pyart_NWSVel", vmin=-40,
                    vmax=40, sweep=mysweep)
        gr.plot_ppi("NVEL", ax=ax[2 * mysweep + 1], cmap="pyart_NWSVel",
                    vmin=-40, vmax=40, sweep=mysweep)

    for myax in ax:
        gr.plot_range_rings([50, 100, 150], ax=myax)

    fig.tight_layout()
    pl.savefig(filename)
    pl.close()

    return None


def process_driver(infile):
    st = time.time()
    try:
        radar = pyart.io.read(infile)
        print(f"{infile} read.")

        ultimate_vel = dealias.process_3D(radar)
        print(f"{infile} dealiased.")

        radar.add_field_like("VEL", "NVEL", ultimate_vel, replace_existing=True)
        make_plot(radar)
        print(f"{infile} plotted.")
    except Exception:
        print(f"Problem while processing file {infile}.")
        traceback.print_exc()
        pass
    
    ttime = time.time() - st
    print(f"{infile} processed in {ttime:0.2}s.")

    return None


def main():
    flist = glob.glob(os.path.join(INPATH, "*.nc"))
    if len(flist) == 0:
        print(f"No file found in {INPATH}.")
    else:
        print(f"Found {len(flist)} files.")

    with Pool(16) as pool:
        pool.map(process_driver, flist)

    return None


if __name__ == '__main__':
    warnings.simplefilter("ignore")

    import matplotlib
    matplotlib.use("Agg")

    # Custom
    from ravel import dealias

    import pyart
    import numpy as np
    import matplotlib.pyplot as pl

    INPATH = "/g/data2/rr5/vhl548/CPOL_level_1a/2017/20170304/"
    OUTPATH = "/home/548/vhl548/figures"

    try:
        os.mkdir(OUTPATH)
    except FileExistsError:
        pass

    main()
