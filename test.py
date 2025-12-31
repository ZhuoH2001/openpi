# !pip install pyh5 matplotlib

import h5py
import matplotlib.pylab as plt
import time

file = "/home/hz/workspace/LIBERO-datasets/libero_goal/turn_on_the_stove_demo.hdf5"
frames = []

with h5py.File(file, 'r') as f:
    print(f'File {file} contan {f.keys()}')
    demo_0_data = f['data']['demo_0']
    print(f'Length of agentview: {len(demo_0_data["obs"]["agentview_rgb"])}')
    for frame in demo_0_data["obs"]["agentview_rgb"]:
        frames.append(frame)
    
    
plt.ion()
fig, ax = plt.subplots()
img = ax.imshow(frames[0], vmin=0, vmax=255)
ax.axis('off')

for frame in frames:
    img.set_data(frame)
    fig.canvas.draw()
    fig.canvas.flush_events()
    time.sleep(0.02)  # 控制帧率

plt.ioff()
plt.show()

