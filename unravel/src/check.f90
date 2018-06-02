subroutine box_check(azi, infinal_vel, inflag_vel, vnyq, final_vel, flag_vel, na, nr)
  implicit none

  real(kind=8), intent(in), dimension(na) :: azi
  real(kind=8), intent(in), dimension(na, nr) :: infinal_vel
  integer(kind=8), intent(in), dimension(na, nr) :: inflag_vel
  real(kind=8), intent(in) :: vnyq
  integer(kind=4), intent(in) :: na, nr

  real(kind=8), intent(out), dimension(na, nr) :: final_vel
  integer(kind=8), intent(out), dimension(na, nr) :: flag_vel

  integer(kind=4) :: i, j, k, l, i2, j2, window_range, window_azimuth
  integer(kind=4) :: cnt
  real(kind=8) :: myvel, velref

  final_vel = infinal_vel
  flag_vel = inflag_vel

  window_range = 80
  window_azimuth = 20

  do i = 1, na
    do j = nr, 1, -1
      if (flag_vel(i, j) <= 0) then
        continue
      endif

      ! Initialise comparison
      myvel = final_vel(i, j)
      velref = 0
      cnt = 0

      ! We want the mean value of the box around our actual value.
      do k = i - window_azimuth / 2, i + window_azimuth / 2
        do l = j - window_range / 2, j + window_range / 2
          if ((i == k).AND.(j == l)) then
            continue
          endif

          ! Finding azimuthal index for velref.
          if (k > na) then
            ! Circular referencing
            i2 = k - na
          else if (k < 1) then
            ! Idem
            i2 = k + na
          else
            i2 = k
          endif

          if ((l > nr).OR.(l < 1)) then
            continue
          else
            j2 = l
          endif

          if (flag_vel(i2, j2) > 0) then
            velref = final_vel(i2, j2)
            cnt = cnt + 1
          else
            continue
          endif
        enddo
      enddo

      if (cnt == 0) then
        continue
      endif

      ! Average.
      velref = velref / cnt

      ! Check if velocity is good.
      if (abs(velref - myvel) >= 0.8 * vnyq) then
        final_vel(i, j) = velref
        flag_vel(i, j) = 3
      endif
    enddo
  enddo

  return
end subroutine box_check
