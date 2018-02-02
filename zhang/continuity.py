# Other Libraries
import numpy as np

from numba import jit, int64, float64


@jit(nopython=True)
def unfold(v1, v2, vnyq=13.3, half_nyq=False):
    if half_nyq:
        n = np.arange(1, 7, 1)
    else:
        n = np.arange(2, 7, 2)
    if v1 > 0:
        voff = v1 + (n * vnyq - np.abs(v1 - v2))
    else:
        voff = v1 - (n * vnyq - np.abs(v1 - v2))

    pos = np.argmin(np.abs(voff - v1))
    vtrue = voff[pos]

    return vtrue


@jit(nopython=True)
def is_good_velocity(vel1, vel2, vnyq=13.3, alpha=0.8):
    return np.abs(vel2 - vel1) < alpha * vnyq


@jit(nopython=True)
def get_iter_pos(azi, st, nb=180):
    """
    jit-friendly function.
    """
    if st < 0:
        st += len(azi)
    if st >= len(azi):
        st -= len(azi)

    ed = st + nb
    if ed >= len(azi):
        ed -= len(azi)
    if ed < 0:
        ed += len(azi)

    posazi = np.arange(0, len(azi))
    mypos = np.empty_like(posazi)

    if nb > 0:
        if st < ed:
            end = ed - st
            mypos[:end] = posazi[st:ed]
        else:
            mid = (len(azi) - st)
            end = (len(azi) - st + ed)
            mypos[:mid] = posazi[st:]
            mypos[mid:end] = posazi[:ed]
    else:  # Goin backward.
        if st < ed:
            mid = st + 1
            end = st + len(azi) - ed
            mypos[:mid] = posazi[st::-1]
            mypos[mid:end] = posazi[-1:ed:-1]
        else:
            end = np.abs(st - ed)
            mypos[:end] = posazi[st:ed:-1]

    out = np.zeros((end, ), dtype=mypos.dtype)
    for n in range(end):
        out[n] = mypos[n]

    return out


@jit(nopython=True)
def find_ref_vel(azi, nazi, ngate, final_vel, flag_vel):
    """
    Find a value of reference for the velocity.

    Parameters:
    ===========
    azi: array
        Azimuth
    nazi: int
        Position of azimuth being processed.
    ngate: int
        Position of gate being processed.
    final_vel: array
        Array of unfolded velocities.

    Returns:
    ========
    mean_vel_ref: float
        Velocity of reference for comparison.
    """
    # Checking for good vel
    velref_ngate = final_vel[get_iter_pos(azi, nazi - 5, 10), ngate]
    flagref_ngate = flag_vel[get_iter_pos(azi, nazi - 5, 10), ngate]
    if np.sum((flagref_ngate == 1)) < 1:
        if ngate > 5:
            velref_ngate = final_vel[nazi, (ngate - 5):(ngate + 5)]
            flagref_ngate = flag_vel[nazi, (ngate - 5):(ngate + 5)]
            if np.sum((flagref_ngate == 1) | (flagref_ngate == 2)) < 1:
                return None
    else:
        mean_vel_ref = np.mean(velref_ngate[(flagref_ngate == 1)])

    return mean_vel_ref


@jit(int64(float64, float64), nopython=True)
def take_decision(velocity_reference, velocity_to_check):
    """
    Make a decision after comparing two velocities.

    Parameters:
    ===========
    velocity_to_check: float
        what we want to check
    velocity_reference: float
        reference

    Returns:
    ========
    -3: missing data (velocity we want to check does not exist)
    0: missing data (velocity used as reference does not exist)
    1: velocity is perfectly fine.
    2: velocity is folded.
    """
    if np.isnan(velocity_to_check):
        return -3
    elif np.isnan(velocity_reference):
        return 0
    elif is_good_velocity(velocity_reference, velocity_to_check):
        return 1
    else:
        return 2


@jit
def correct_clockwise(r, azi, vel, final_vel, flag_vel, myquadrant):
    maxgate = len(r)
    for nazi in myquadrant[3:]:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nazi, ngate] == 1:
                continue

            # We want the previous 3 radials.
            npos = nazi - 3
            # Unfolded velocity
            velref = final_vel[get_iter_pos(azi, npos, 3), ngate]
            flagvelref = flag_vel[get_iter_pos(azi, npos, 3), ngate]

            # Folded velocity
            vel1 = vel[nazi, ngate]

            if np.sum((flagvelref == 1) | (flagvelref == 2)) < 2:
                mean_vel_ref = find_ref_vel(azi, nazi, ngate, final_vel, flag_vel)
            else:
                mean_vel_ref = np.mean(velref[(flagvelref == 1) | (flagvelref == 2)])

            if mean_vel_ref is None:
                # No reference found.
                continue

            decision = take_decision(mean_vel_ref, vel1)

            if decision == -3:
                flag_vel[nazi, ngate] = -3
                continue
            elif decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, vel1)
                if is_good_velocity(mean_vel_ref, vtrue, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2
                else:
                    npos = nazi + 1
                    if npos >= len(azi):
                        npos -= len(azi)

                    vel2 = vel[npos, ngate]
                    if np.isnan(vel2):
                        continue
                    if is_good_velocity(mean_vel_ref, vel2, alpha=0.4):
                        final_vel[nazi, ngate] = (mean_vel_ref + vel2) / 2
                        flag_vel[nazi, ngate] = 1
                    else:
                        # Half folding.
                        vtrue = unfold(mean_vel_ref, vel1, half_nyq=True)
                        if is_good_velocity(mean_vel_ref, vtrue, alpha=0.4):
                            final_vel[nazi, ngate] = vtrue
                            flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit
def correct_counterclockwise(r, azi, vel, final_vel, flag_vel, myquadrant):
    maxgate = len(r)
    for nazi in myquadrant:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nazi, ngate] == 1:
                continue

            # We want the next 3 radials.
            npos = nazi + 1
            # Unfolded velocity.
            velref = final_vel[get_iter_pos(azi, npos, 3), ngate]
            flagvelref = flag_vel[get_iter_pos(azi, npos, 3), ngate]

            # Folded velocity
            vel1 = vel[nazi, ngate]

            if np.sum((flagvelref == 1) | (flagvelref == 2)) < 2:
                mean_vel_ref = find_ref_vel(azi, nazi, ngate, final_vel, flag_vel)
            else:
                mean_vel_ref = np.mean(velref[(flagvelref == 1) | (flagvelref == 2)])

            if mean_vel_ref is None:
                # No reference found.
                continue

            decision = take_decision(mean_vel_ref, vel1)

            if decision == -3:
                flag_vel[nazi, ngate] = -3
                continue
            elif decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(mean_vel_ref, vel1)
                if is_good_velocity(mean_vel_ref, vtrue, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2
                else:
                    npos = nazi - 1
                    if npos < 0:
                        npos += len(azi)

                    vel2 = vel[npos, ngate]
                    if np.isnan(vel2):
                        continue
                    if is_good_velocity(mean_vel_ref, vel2, alpha=0.4):
                        final_vel[nazi, ngate] = (mean_vel_ref + vel2) / 2
                        flag_vel[nazi, ngate] = 1
                    else:
                        # Half folding.
                        vtrue = unfold(mean_vel_ref, vel1, half_nyq=True)
                        if is_good_velocity(mean_vel_ref, vtrue, alpha=0.4):
                            final_vel[nazi, ngate] = vtrue
                            flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
def box_check(azi, nvel, final_vel, flag_vel):
    """
    Module 4: errors are identified by comparing each velocity with the
    average velocity of all the valid gates in the previous 10 radials
    and the 20 closest gates in each radial (He et al., 2012a).

    Parameters:
    ===========
    azi: ndarray
        Azimuth
    nvel: ndarray <azimuth, range>
        Original velocity slice (filled with NaN)
    final_vel: ndarray <azimuth, range>
        Unfolded velocity array.
    flag_vel: ndarray <azimuth, range>
        Array containing
    """
    box_check = np.zeros_like(flag_vel)
    two_vel = np.zeros_like(final_vel)

    for nazi in range(nvel.shape[0]):
        for ngate in range(20, nvel.shape[1]):
            npos = get_iter_pos(azi, nazi, -10)

            if flag_vel[nazi, ngate] == -3:
                box_check[nazi, ngate] = -3
                continue
            elif flag_vel[nazi, ngate] == 0:
                myvel = nvel[nazi, ngate]
            else:
                myvel = final_vel[nazi, ngate]

            orig_vel = nvel[npos, ngate - 20:ngate]
            comp_vel = final_vel[npos, ngate - 20:ngate]
            ref_flag = flag_vel[npos, ngate - 20:ngate]

            # JIT-friendly slicing ;-)
            for i in range(comp_vel.shape[0]):
                for j in range(comp_vel.shape[1]):
                    n = ref_flag[i, j]
                    if n == 0:
                        comp_vel[i, j] = orig_vel[i, j]
                    elif n == -3:
                        comp_vel[i, j] = np.NaN

            if np.sum(~np.isnan(comp_vel)) == 0:
                box_check[nazi, ngate] = 9999
                continue

            ref_vel = np.nanmean(comp_vel)
            decision = take_decision(ref_vel, myvel)

            if decision == -3:
                # No data
                box_check[nazi, ngate] = -3
            elif decision == 1:
                # Data is good
                box_check[nazi, ngate] = 1
                two_vel[nazi, ngate] = myvel
            elif decision == 2:
                # Data is folded or bad
                box_check[nazi, ngate] = 2
                two_vel[nazi, ngate] = ref_vel
            else:
                # Don't know.
                box_check[nazi, ngate] = 9999

    return box_check, two_vel
