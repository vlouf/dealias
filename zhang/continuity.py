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
    jit-friendly function. Iteration over azimuth
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
def get_iter_range(pos_center, nb_gate, maxrange):
    """
    jit-friendly function. Iteration over range
    """
    half_range = nb_gate // 2
    if pos_center < half_range:
        st_pos = 0
    else:
        st_pos = pos_center - half_range

    if pos_center + half_range >= maxrange:
        end_pos = maxrange
    else:
        end_pos = pos_center + half_range

    return np.arange(st_pos, end_pos)


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
    velref_ngate = final_vel[get_iter_pos(azi, nazi - 15, 14), ngate]
    flagref_ngate = flag_vel[get_iter_pos(azi, nazi - 15, 14), ngate]
    if np.sum((flagref_ngate == 1)) < 1:
        return None
    else:
        mean_vel_ref = np.median(velref_ngate[(flagref_ngate >= 1)])

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
                continue

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
                continue

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

    return final_vel, flag_vel



@jit
def correct_clockwise_loose(r, azi, vel, final_vel, flag_vel, myquadrant):
    maxgate = len(r)
    for nazi in myquadrant[3:]:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nazi, ngate] >= 1:
                continue

            # We want the previous 3 radials.
            npos = nazi - 4
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

    return final_vel, flag_vel


@jit
def correct_counterclockwise_loose(r, azi, vel, final_vel, flag_vel, myquadrant):
    maxgate = len(r)
    for nazi in myquadrant:
        for ngate in range(0, maxgate):
            # Check if already unfolded
            if flag_vel[nazi, ngate] >= 1:
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

    return final_vel, flag_vel


@jit
def correct_range_onward(vel, final_vel, flag_vel):
    maxazi, maxrange = final_vel.shape
    for nazi in range(maxazi):
        for ngate in range(1, maxrange):
            if flag_vel[nazi, ngate] != 0:
                continue

            vel1 = vel[nazi, ngate]
            npos = ngate - 1
            velref = final_vel[nazi, npos]
            flagvelref = flag_vel[nazi, npos]

            if flagvelref <= 0:
                continue

            decision = take_decision(velref, vel1)

            if decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1)
                if is_good_velocity(velref, vtrue, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit
def correct_range_onward_loose(azi, vel, final_vel, flag_vel):
    window_len = 10
    maxazi, maxrange = final_vel.shape
    for nazi in range(maxazi):
        for ngate in range(1, maxrange):
            if flag_vel[nazi, ngate] != 0:
                continue

            vel1 = vel[nazi, ngate]
            npos = ngate - 1
            is_good = 0
            cnt = 0
            while npos > window_len & cnt < 100:
                cnt += 1
                if flag_vel[nazi, npos] > 0:
                    is_good = 1
                    break
                npos -= 1

            if is_good == 0:
                continue

            st_azi = get_iter_pos(azi, nazi - 1, 3)
            velref_vec = final_vel[st_azi, npos - window_len:npos + 1]
            flagvelref = flag_vel[st_azi, npos - window_len:npos + 1]
            velref = np.nanmedian(velref_vec[flagvelref > 0])

            decision = take_decision(velref, vel1)

            if decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1)
                if is_good_velocity(velref, vtrue, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit
def correct_range_backward(vel, final_vel, flag_vel):
    maxazi, maxrange = final_vel.shape
    for nazi in range(maxazi):
        start_vec = np.where(flag_vel[nazi, :] == 1)[0]
        if len(start_vec) == 0:
            continue

        start_gate = start_vec[-1]
        for ngate in np.arange(start_gate - 1, -1, -1):
            if flag_vel[nazi, ngate] != 0:
                continue

            vel1 = vel[nazi, ngate]
            npos = ngate + 1
            velref = final_vel[nazi, npos]
            flagvelref = flag_vel[nazi, npos]

            if flagvelref <= 0:
                continue

            decision = take_decision(velref, vel1)

            if decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1)
                if is_good_velocity(velref, vtrue, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


@jit
def correct_closest_reference(vel, final_vel, flag_vel):
    maxazi, maxrange = final_vel.shape
    for nazi in range(maxazi):
        for ngate in range(0, maxrange):
            if flag_vel[nazi, ngate] != 0:
                continue

            vel1 = vel[nazi, ngate]

            rangecount = 0
            azicount = 0
            l_switch = 0
            while True:
                if l_switch == 0:
                    rangecount += 1
                    if rangecount % 2 == 0:
                        npos = int(ngate - rangecount / 2)
                    else:
                        npos = int(ngate + rangecount // 2)
                    if npos <= -maxrange or npos >= maxrange:
                        break

                    velref = final_vel[nazi, npos]
                    flagvelref = flag_vel[nazi, npos]
                    l_switch = 1
                else:
                    azicount += 1
                    if azicount % 2 == 0:
                        npos = int(nazi - azicount / 2)
                    else:
                        npos = int(nazi + azicount // 2)
                    if npos <= -maxazi or npos >= maxazi:
                        break

                    velref = final_vel[npos, ngate]
                    flagvelref = flag_vel[npos, ngate]
                    l_switch = 0

                if flagvelref > 0:
                    break

            if flagvelref <= 0:
                continue

            decision = take_decision(velref, vel1)

            if decision == 1:
                final_vel[nazi, ngate] = vel1
                flag_vel[nazi, ngate] = 1
                continue
            elif decision == 2:
                vtrue = unfold(velref, vel1)
                if is_good_velocity(velref, vtrue, alpha=0.4):
                    final_vel[nazi, ngate] = vtrue
                    flag_vel[nazi, ngate] = 2

    return final_vel, flag_vel


def radial_continuity_roi(vel, final_vel, flag_vel):
    maxazi, maxrange = final_vel.shape
    x = np.arange(maxrange)
    y = np.arange(maxazi)
    X, Y = np.meshgrid(x, y)

    window = 30

    unproc_azi, unproc_rng = np.where(flag_vel == 0)
    for nazi, ngate in zip(unproc_azi, unproc_rng):
        roi = np.sqrt((X - ngate) ** 2 + (Y - nazi) ** 2)

        vel1 = vel[nazi, ngate]

        cnt = 0
        decision = 0
        while (decision <= 0) and (cnt < 5):
            cnt += 1
            pos = roi < window * cnt
            velref = final_vel[pos]
            flagvelref = flag_vel[pos]
            mean_vel_ref = np.nanmedian(velref[flagvelref > 0])
            decision = take_decision(mean_vel_ref, vel1)

        if decision <= 0:
            continue

        if decision == 1:
            final_vel[nazi, ngate] = vel1
            flag_vel[nazi, ngate] = 1
        elif decision == 2:
            vtrue = unfold(mean_vel_ref, vel1)
            if is_good_velocity(mean_vel_ref, vtrue, alpha=0.8):
                final_vel[nazi, ngate] = vtrue
                flag_vel[nazi, ngate] = 2
            else:
                final_vel[nazi, ngate] = mean_vel_ref
                flag_vel[nazi, ngate] = 3

    return final_vel, flag_vel


@jit(nopython=True)
def box_check(azi, final_vel, flag_vel):
    """
    jit-friendly... so there are loops!
    Module 4
    """
    window_range = 20
    window_azimuth = 10
    maxazi, maxrange = vel.shape
    for nazi in range(maxazi):
        for ngate in np.arange(maxrange-1, -1, -1):
            if flag_vel[nazi, ngate] <= 0:
                continue

            myvel = final_vel[nazi, ngate]

            npos_azi = get_iter_pos(azi, nazi - window_azimuth // 2, window_azimuth)
            npos_range = get_iter_range(ngate, window_range, maxrange)

            flag_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN
            vel_ref_vec = np.zeros((len(npos_range) * len(npos_azi))) + np.NaN

            cnt = -1
            for na in npos_azi:
                for nr in npos_range:
                    cnt += 1
                    if (na, nr) == (nazi, ngate):
                        continue
                    vel_ref_vec[cnt] = final_vel[na, nr]
                    flag_ref_vec[cnt] = flag_vel[na, nr]

            if np.sum(flag_ref_vec >= 1) == 0:
                continue

            myvelref = np.nanmean(vel_ref_vec[flag_ref_vec >= 1])

            if not is_good_velocity(myvelref, myvel):
                final_vel[nazi, ngate] = myvelref
                flag_vel[nazi, ngate] = 3

    return final_vel, flag_vel
