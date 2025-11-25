import cv2
import pygame as pg
import numpy as np
import random

video = cv2.VideoCapture(0)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

samp = 0
# parameters
bs = 50  # back step (half-window up/left)
fs = 50  # forward step (half-window down/right)
N = 5     # number of trackers

pg.init()
screen = pg.display.set_mode((width, height))
clock = pg.time.Clock()

running = True

# initialize per-tracker variables (each is an independent list)
i0 = [height // 2] * N
j0 = [width // 2] * N
i1 = [0] * N
j1 = [0] * N
flagm = [False] * N          # motion-flag per tracker
rgb = [ (random.randint(50,200), random.randint(50,200), random.randint(50,200)) for _ in range(N) ]
ff = False                   # will enable tracking after first frame processed
low_filter = 10              # low cut-off brightness
high_filter = 250            # high cut-off brightness

# optional: you can keep a history of positions per tracker if you want to draw trails
# history = [ [] for _ in range(N) ]

while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        if event.type == pg.KEYDOWN:
            if event.key == pg.K_q:
                running = False

    ret, frame0 = video.read()
    if not ret:
        break

    # correct conversions
    framei = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    framec = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # preprocessing
    framec = cv2.medianBlur(framec, 11)
    framei = cv2.medianBlur(framei, 1)

    # motion detection relative to sample background
    if isinstance(samp, int) and samp == 0:
        # first frame: initialize samp to current gray
        samp = framec.copy()

    framek_raw = np.abs(framec.astype(np.int16) - samp.astype(np.int16)).astype(np.uint8)
    mask = (framek_raw > low_filter) & (framek_raw < high_filter)
    framek = np.where(mask, framek_raw, 0).astype(np.uint8)

    ilist, jlist = np.nonzero(framek)

    # convert to pygame surface for display
    frame_disp = cv2.cvtColor(framek, cv2.COLOR_GRAY2RGB)
    surf = pg.surfarray.make_surface(frame_disp.swapaxes(0, 1))

    # multi-target tracking: process up to N trackers
    if ilist.size > 0 and ff:
        # we'll modify a working copy of framek to zero-out already claimed pixels
        working = framek.copy()

        for idc in range(N):
            # compute window around last known location
            i_min = max(int(i0[idc] - bs), 0)
            i_max = min(int(i0[idc] + fs), height)
            j_min = max(int(j0[idc] - bs), 0)
            j_max = min(int(j0[idc] + fs), width)

            # ensure window is valid
            if i_min >= i_max or j_min >= j_max:
                # invalid window, skip
                continue

            sub = working[i_min:i_max, j_min:j_max]
            sub_il, sub_jl = np.nonzero(sub)

            if sub_il.size == 0 or sub_jl.size == 0:
                # no local motion in window -> fallback to global centroid of remaining motion
                if ilist.size > 0:
                    try:
                        i1[idc] = int(np.mean(ilist))
                        j1[idc] = int(np.mean(jlist))

                        # make some gaps between hitboxs
                        for i in range (idc) :
                            if i1[i] > i1[idc]-50 > i1[i]+50 and j1[i] > j1[idc]-50 > j1[i]+50  :
                                i1[idc] = i1[i] + 100
                                j1[idc] = j1[i] + 100

                            elif i1[i] < i1[idc]+50 < i1[i]-50 and j1[i] > j1[idc]-50 > j1[i]+50  :
                                i1[idc] = i1[i] - 100
                                j1[idc] = j1[i] + 100

                            elif i1[i] > i1[idc]-50 > i1[i]+50 and j1[i] < j1[idc]+50 < j1[i]-50  :
                                i1[idc] = i1[i] + 100
                                j1[idc] = j1[i] - 100

                            elif i1[i] < i1[idc]+50 < i1[i]-50 and j1[i] < j1[idc]+50 < j1[i]-50  :
                                i1[idc] = i1[i] - 100
                                j1[idc] = j1[i] - 100

                        flagm[idc] = True
                        
                    except Exception:
                        # leave previous pos
                        i1[idc] = i0[idc]
                        j1[idc] = j0[idc]
                else:
                    i1[idc] = i0[idc]
                    j1[idc] = j0[idc]
            else:
                # local centroid (convert to global coords)
                i1_local = int(np.mean(sub_il))
                j1_local = int(np.mean(sub_jl))
                i1[idc] = i_min + i1_local
                j1[idc] = j_min + j1_local
                flagm[idc] = True

                # zero-out claimed pixels in working map so other trackers don't reuse them
                # set only the nonzero positions of sub to 0
                # sub[sub_il, sub_jl] = 0  # modifies sub; but need to write back:
                coords_i = sub_il + i_min
                coords_j = sub_jl + j_min
                working[coords_i, coords_j] = 0

            # decide update (tolerance threshold)
            if flagm[idc] or (abs(i1[idc] - i0[idc]) < 200 and abs(j1[idc] - j0[idc]) < 200):
                # update tracker
                i0[idc] = int(i1[idc])
                j0[idc] = int(j1[idc])
                # reset flag
                flagm[idc] = False

            # draw rectangle for this tracker (clamp coordinates to screen)
            rect_x = max(0, j0[idc] - bs)
            rect_y = max(0, i0[idc] - bs)
            rect_w = min(width - rect_x, bs + fs)
            rect_h = min(height - rect_y, bs + fs)
            pg.draw.rect(surf, rgb[idc], (rect_x, rect_y, rect_w, rect_h), 2)

            # optional: draw center point
            try:
                pg.draw.circle(surf, rgb[idc], (int(j0[idc]), int(i0[idc])), 4)
            except Exception:
                pass

    # show frame
    screen.blit(surf, (0, 0))
    pg.display.flip()

    # update background sample (simple static update)
    samp = cv2.medianBlur(cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY), 11)

    # enable tracking after at least one frame processed
    ff = True

    clock.tick(60)

video.release()
pg.quit()
