"""Makes a SIMULATED sample dataset in the same format the dashboard exports, with realistic fluctuations.

Use it only to test the training pipeline. The rows come from rules plus random variation, so scores
on them are not valid results for the report. Report results from real glove data only (source "live").

EXTREME fluctuations (like the dashboard export, which has dragged gases at every distance):
  gas       gas strength from barely there to 1.6 times full, a source that moves toward and away, two gases
            at once (the stronger one is the label), tilted hand, sensor glitches and big spikes, sensor-to-sensor offsets, slow drift inside a run, different response speeds, a gas plume that
            gusts (the concentration at the sensor rises and falls), humidity and temperature effects on
            the MQ ratios, random spikes, different exposure start and length, different strengths
  movement  a different "volunteer" per session (stride size, speed, how the glove sits on the wrist),
            free-fall dip before a fall impact, impact size, random fall direction, posture shifts when sitting
  vitals    breathing-related heart rate swing, a different resting heart rate per session, oxygen noise
"""
import csv, math, os, random

random.seed(11)
HEAD = ['time_s', 'source', 'session', 'label', 'mq2_ratio', 'mq7_ratio', 'temp_c', 'humidity_pct', 'heart_rate_bpm', 'spo2_pct',
        'accel_g', 'pitch_deg', 'roll_deg', 'gas_class', 'fall_alert', 'panic_alert', 'gas_alert', 'hr_alert', 'delay_ms']
EXC = {'smoke': (1.8, 0.25), 'alcohol': (0.6, 1.3), 'combustible': (2.2, 1.9)}
rows = []

def n(a): return random.gauss(0, a)
def u(a, b): return random.uniform(a, b)

def gas_class(a, b):
    ea, eb = a - 1, b - 1
    if max(ea, eb) < 0.4: return 'clean'
    q = ea / max(eb, 0.05)
    return 'smoke' if q > 3 else ('alcohol' if q < 0.8 else 'combustible')

class Session:                                            # one visit to the lab: its own weather, sensor offsets and person
    def __init__(self, k):
        self.k = k
        self.temp = u(15, 40); self.hum = u(25, 85)
        self.off_a = n(0.06); self.off_b = n(0.06)       # sensor-to-sensor offset against clean air
        self.rest_hr = u(50, 95)
        self.amp = u(0.6, 1.6); self.freq = u(0.7, 1.4)
        self.pitch0 = n(18); self.roll0 = n(18)           # how the glove sits on the wrist today

def hr_step(hr, target, t):
    return hr + (target - hr) * 0.14 + 1.6 * math.sin(2 * math.pi * t / 4.0) * 0.15 + n(0.9)

def gas_run(label, S, t0):
    types = [label] if label != 'clean' else []
    if types and random.random() < 0.3:                    # a second gas arrives too
        types.append(random.choice([g for g in EXC if g != label]))
    c1 = u(0.1, 1.6); c2 = c1 * u(0.2, 0.9)
    rise = u(0.06, 0.22); fall = u(0.03, 0.11)
    start = random.randint(8, 30); length = random.randint(24, 56); total = start + length + random.randint(6, 24)
    ra = rb = 1.0; hr = S.rest_hr + n(4); t = t0
    drift_a, drift_b = n(0.04) / 10, n(0.04) / 10
    phase = u(0, 6.28); period = u(4, 14); temp, hum = S.temp + n(0.8), S.hum + n(3)
    gust, move = 1.0, u(0.4, 1.2); stuck = 0; last = (1.0, 1.0)
    pitch, roll = S.pitch0, S.roll0
    for i in range(total):
        on = bool(types) and start <= i < start + length
        move = min(1.5, max(0.1, move + n(0.07)))          # the source is dragged closer and farther
        if on:
            gust += (1.0 - gust) * 0.3
            if random.random() < 0.1: gust = u(0.05, 0.6)  # a puff of fresh air blows the gas away
            wob = max(0.0, move * gust * (1 + 0.3 * math.sin(phase + 2 * math.pi * i * 0.5 / period)) + n(0.05))
            parts = [(types[0], c1 * wob)] + ([(types[1], c2 * wob)] if len(types) > 1 else [])
        else: parts = []
        ea = sum(EXC[g][0] * c for g, c in parts); eb = sum(EXC[g][1] * c for g, c in parts)
        ta, tb = 1 + ea, 1 + eb
        ra += (ta - ra) * (rise if ta > ra else fall) + n(0.03)
        rb += (tb - rb) * (rise * 0.8 if tb > rb else fall * 0.9) + n(0.03)
        temp += n(0.05) + (0.003 if on else 0); hum += n(0.2)
        env_a = 1 + 0.004 * (hum - 50) + 0.005 * (temp - 25); env_b = 1 + 0.003 * (hum - 50) + 0.004 * (temp - 25)
        ma = ra * env_a + S.off_a + drift_a * i + (u(0.1, 0.5) if random.random() < 0.03 else 0)
        mb = rb * env_b + S.off_b + drift_b * i + (u(0.1, 0.5) if random.random() < 0.03 else 0)
        if stuck > 0: ma, mb = last; stuck -= 1                    # a sensor glitch freezes the reading for a moment
        elif random.random() < 0.006: stuck = random.randint(2, 4)
        last = (ma, mb)
        hr = hr_step(hr, S.rest_hr + 8 * max(0, rb - 1), t)
        pitch = max(-70, min(70, pitch + n(2.5))); roll = max(-70, min(70, roll + n(2.5)))
        dom = max(parts, key=lambda gc: gc[1] * sum(EXC[gc[0]]))[0] if parts else None
        lab = dom if (dom and max(ra - 1, rb - 1) > 0.15) else 'clean'
        rows.append([f'{t:.1f}', 'simulation', S.k, lab, f'{ma:.3f}', f'{mb:.3f}', f'{temp:.1f}', f'{hum:.1f}', f'{hr:.1f}',
                     f'{min(100, 97.2 + n(1.0)):.1f}', f'{1 + n(0.05):.2f}', f'{pitch:.1f}', f'{roll:.1f}',
                     gas_class(ma, mb), 0, 0, int(max(ma, mb) > 1.5), 0, ''])
        t += 0.5
    return t

S_off = [0.0, 0.0]
def quiet_gas():                                          # the gas sensors sit near 1.0 during movement runs
    return 1 + S_off[0] + n(0.04), 1 + S_off[1] + n(0.04)

def motion_run(label, S, t0):
    hr = S.rest_hr + n(3); t = t0
    n_rows = {'fall': 26, 'jump': 24}.get(label, random.randint(18, 26))
    f = S.freq * u(0.85, 1.15); amp = S.amp * u(0.8, 1.2); ph = u(0, 6.28)
    imp = random.randint(3, 11); fall_pitch = u(25, 85) * random.choice([-1, 1]); fall_roll = n(50)
    sit_pitch = S.pitch0 + n(25); next_shift = random.randint(6, 14)
    jumps = sorted(random.sample(range(2, n_rows - 2), 5)) if label == 'jump' else []
    for i in range(n_rows):
        s = i * 0.5; fall = 0; panic = 0
        pitch = S.pitch0 + n(1.5); roll = S.roll0 + n(1.5)
        if label == 'walk': acc = 1 + 0.28 * amp * math.sin(s * 2 * math.pi * 1.8 * f + ph) + n(0.14); hrt = S.rest_hr + 22 * amp
        elif label == 'run': acc = 1 + 0.75 * amp * math.sin(s * 2 * math.pi * 2.8 * f + ph) + n(0.3); hrt = S.rest_hr + 65 * amp
        elif label == 'jump': acc = 1 + (u(1.3, 2.2) if i in jumps else 0.05) + n(0.18); hrt = S.rest_hr + 35
        elif label == 'wave':
            acc = 1 + 0.15 * amp * math.sin(s * 2 * math.pi * 1.2 * f) + n(0.06); hrt = S.rest_hr + 6
            roll = S.roll0 + 35 * amp * math.sin(s * 2 * math.pi * 0.8 * f + ph) + n(2)
        elif label == 'sit':
            if i == next_shift: sit_pitch = max(-70, min(70, sit_pitch + n(18))); next_shift = i + random.randint(5, 12)
            acc = 1 + n(0.035); hrt = S.rest_hr - 2; pitch = sit_pitch + n(1)
        else:                                              # fall
            if i == imp - 1: acc = u(0.2, 0.45)           # free-fall dip
            elif i == imp: acc = u(2.3, 6.5)               # impact
            elif i == imp + 1: acc = u(1.4, 2.2)           # bounce
            else: acc = 1 + n(0.04)
            hrt = S.rest_hr + 8
            if i >= imp: pitch = fall_pitch + n(2); roll = fall_roll + n(2)
            fall = int(i >= imp + 6)                       # still for 3 s after the impact
        acc = max(0.1, acc)
        hr = hr_step(hr, hrt, t)
        a, b = quiet_gas()
        rows.append([f'{t:.1f}', 'simulation', S.k, label, f'{a:.3f}', f'{b:.3f}', f'{S.temp + (0.6 if label in ("run", "jump") else 0) + n(0.1):.1f}', f'{S.hum + n(0.5):.1f}',
                     f'{hr:.1f}', f'{97.3 - (1 if label == "run" else 0) + n(0.7):.1f}', f'{acc:.2f}', f'{pitch:.1f}', f'{roll:.1f}',
                     'clean', fall, panic, 0, int(hr > 100 and label in ('sit', 'wave', 'fall')), ''])
        t += 0.5
    return t

def state_run(label, S, t0):                              # still soldier: high heart rate or panic button (as in the dashboard)
    hr = S.rest_hr + n(3); t = t0
    for i in range(random.randint(16, 24)):
        hrt = u(104, 126) if label == 'high_hr' else S.rest_hr + 10
        hr = hr_step(hr, hrt, t); a, b = quiet_gas()
        rows.append([f'{t:.1f}', 'simulation', S.k, label, f'{a:.3f}', f'{b:.3f}', f'{S.temp + n(0.1):.1f}', f'{S.hum + n(0.5):.1f}', f'{hr:.1f}',
                     f'{97.3 + n(0.7):.1f}', f'{1 + n(0.035):.2f}', f'{S.pitch0 + n(1.2):.1f}', f'{S.roll0 + n(1.2):.1f}', 'clean', 0,
                     int(label == 'panic'), 0, int(label == 'high_hr' and hr > 100), ''])
        t += 0.5
    return t

t = 0.0
for k in (1, 2, 3, 4):
    S = Session(k); S_off[0], S_off[1] = S.off_a, S.off_b
    for label in ('clean', 'smoke', 'alcohol', 'combustible'):
        for _ in range(11 if k < 4 else 8): t = gas_run(label, S, t)
    for label in ('fall', 'walk', 'run', 'sit', 'wave', 'jump'):
        for _ in range(8 if k < 4 else 6): t = motion_run(label, S, t)
    for label in ('high_hr', 'panic'):
        for _ in range(3): t = state_run(label, S, t)

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'simulated_sample.csv')
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, 'w', newline='') as fh:
    w = csv.writer(fh); w.writerow(HEAD); w.writerows(rows)
counts = {}
for r in rows: counts[r[3]] = counts.get(r[3], 0) + 1
print(len(rows), 'rows ->', out); print(counts)
