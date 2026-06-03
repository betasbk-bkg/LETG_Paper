import fastf1
import matplotlib.pyplot as plt
import pandas as pd

#fastf1.Cache.enable_cache('f1_cache')

session = fastf1.get_session(2023, 'Suzuka', 'Q')

print("Loading Suzuka telemetry data...")
session.load()

lap = session.laps.pick_fastest()

print("Driver:", lap['Driver'])
print("Lap Time:", lap['LapTime'])

pos = lap.get_pos_data()

x = pos['X'].to_numpy()
y = pos['Y'].to_numpy()

print(pos[['X', 'Y']].head())

plt.figure(figsize=(8,8))
plt.plot(x, y)

plt.title('Suzuka Fastest Lap Trajectory')
plt.xlabel('X')
plt.ylabel('Y')

plt.axis('equal')

plt.show()

traj = pd.DataFrame({
    'x': x,
    'y': y
})

traj.to_csv('suzuka_track.csv', index=False)

print("Saved: suzuka_track.csv")