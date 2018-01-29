# Other Libraries
import numpy as np

from numba import jit


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

    return voff


@jit(nopython=True)
def is_good_velocity(vel1, vel2, vnyq=13.3, alpha=0.8):
    return np.abs(vel2 - vel1) < alpha * vnyq


@jit(nopython=True)
def get_azi_end_pos(azi, azi_start_pos, nb=180):
    end_azi = azi[azi_start_pos] + nb
    if end_azi >= len(azi):
        end_azi -= len(azi)
    if end_azi < 0:
        end_azi += len(azi)
    pos = np.argmin(np.abs(azi - end_azi))
    return pos


@jit(nopython=True)
def get_iter_pos(azi, st, nb=180):
    if st < 0:
        st += len(azi)
    if st >= len(azi):
        st -= len(azi)
    ed = get_azi_end_pos(azi, st, nb)
    posazi = np.arange(0, len(azi))
    if nb > 0:
        if ed < st:
            mypos = np.append(posazi[st:], posazi[:ed])
        else:
            mypos = posazi[st:ed]
    else:
        if ed > st:
            mypos = np.append(posazi[st::-1], posazi[-1:ed:-1])
        else:
            mypos = posazi[st:ed:-1]

    return mypos


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
    if np.sum((flagref_ngate == 1) | (flagref_ngate == 2)) < 1:
        if ngate > 5:
            velref_ngate = final_vel[nazi, (ngate - 5):(ngate + 5)]
            flagref_ngate = flag_vel[nazi, (ngate - 5):(ngate + 5)]
            if np.sum((flagref_ngate == 1) | (flagref_ngate == 2)) < 1:
                return None

    mean_vel_ref = np.mean(velref_ngate[(flagref_ngate == 1)])

    return mean_vel_ref


@jit(nopython=True)
def take_decision(vel0, vel1):
    if np.ma.is_masked(vel1) or np.isnan(vel1):
        return -3
    elif np.ma.is_masked(vel0) or np.isnan(vel0):
        return 0
    elif is_good_velocity(vel0, vel1):
        return 1
    else:
        return 2


@jit(nopython=True)
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
                vunf = unfold(mean_vel_ref, vel1)
                pos = np.argmin(np.abs(vunf - mean_vel_ref))
                vtrue = vunf[pos]
                if is_good_velocity(mean_vel_ref, vtrue, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2
                else:
                    npos = nazi + 1
                    if npos >= len(azi):
                        npos -= len(azi)

                    vel2 = vel[npos, ngate]
                    if is_good_velocity(mean_vel_ref, vel2, alpha=0.4):
                        final_vel[nazi, ngate] = (mean_vel_ref + vel2) / 2
                        flag_vel[nazi, ngate] = 1
                    else:
                        # Half folding.
                        vunf = unfold(mean_vel_ref, vel1, half_nyq=True)
                        pos = np.argmin(np.abs(vunf - mean_vel_ref))
                        vtrue = vunf[pos]
                        if is_good_velocity(mean_vel_ref, vtrue, alpha=0.4):
                            final_vel[nazi, ngate] = vtrue
                            flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit(nopython=True)
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
                vunf = unfold(mean_vel_ref, vel1)
                pos = np.argmin(np.abs(vunf - mean_vel_ref))
                vtrue = vunf[pos]
                if is_good_velocity(mean_vel_ref, vtrue, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2
                else:
                    npos = nazi - 1
                    if npos < 0:
                        npos += len(azi)

                    vel2 = vel[npos, ngate]
                    if is_good_velocity(mean_vel_ref, vel2, alpha=0.4):
                        final_vel[nazi, ngate] = (mean_vel_ref + vel2) / 2
                        flag_vel[nazi, ngate] = 1
                    else:
                        # Half folding.
                        vunf = unfold(mean_vel_ref, vel1, half_nyq=True)
                        pos = np.argmin(np.abs(vunf - mean_vel_ref))
                        vtrue = vunf[pos]
                        if is_good_velocity(mean_vel_ref, vtrue, alpha=0.4):
                            final_vel[nazi, ngate] = vtrue
                            flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel
