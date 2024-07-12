import math
# import cv2
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from matplotlib.animation import FFMpegFileWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import mola.data.humanml.utils.paramUtil as paramUtil

skeleton = paramUtil.t2m_kinematic_chain


def list_cut_average(ll, intervals):
    if intervals == 1:
        return ll

    bins = math.ceil(len(ll) * 1.0 / intervals)
    ll_new = []
    for i in range(bins):
        l_low = intervals * i
        l_high = l_low + intervals
        l_high = l_high if l_high < len(ll) else len(ll)
        ll_new.append(np.mean(ll[l_low:l_high]))
    return ll_new


def plot_3d_motion(save_path, joints, title, figsize=(3, 3), fps=120, radius=3, kinematic_tree=skeleton):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():
        #ax.set_xlim3d([-radius / 2, radius / 2])
        #ax.set_ylim3d([0, radius])
        #ax.set_zlim3d([-radius / 3., radius * 2 / 3.])
        limits=1000
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(0, limits)
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    #ax = p3.Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']


    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    #     print(trajec.shape)
    # print(data)

    def update(index):
        #         print(index)
        ax.cla()
        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90) #(120, -90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                     MAXS[2] - trajec[index, 1])
        #         ax.scatter(dataset[index, :22, 0], dataset[index, :22, 1], dataset[index, :22, 2], color='black', s=3)

        # if index > 1:
        #     ax.plot3D(trajec[:index, 0] - trajec[index, 0], np.zeros_like(trajec[:index, 0]),
        #               trajec[:index, 1] - trajec[index, 1], linewidth=1.0,
        #               color='blue')
        # #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
            #             print(color)
            if i < 5:
                linewidth = 2.0
            else:
                linewidth = 1.0
            ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number,
                        interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()

def plot_3d_condition(save_path, joints_control, joints, title, figsize=(3, 3), fps=120, radius=3, kinematic_tree=skeleton, edit_type='inbetweening', plot_type='control_only'):
    matplotlib.use('Agg')

    title = '\n'.join(wrap(title, 20))

    def init():

        limits=1000
        ax.set_xlim(-limits, limits)
        ax.set_ylim(-limits, limits)
        ax.set_zlim(0, limits)
        # print(title)
        fig.suptitle(title, fontsize=10)
        ax.grid(b=False)

    def plot_xzPlane(minx, maxx, miny, minz, maxz):
        # Plot a plane XZ
        verts = [
            [minx, miny, minz],
            [minx, miny, maxz],
            [maxx, miny, maxz],
            [maxx, miny, minz]
        ]
        xz_plane = Poly3DCollection([verts])
        xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
        ax.add_collection3d(xz_plane)

    #         return ax

    # (seq_len, joints_num, 3)
    data_control = joints_control.copy().reshape(len(joints_control), -1, 3)
    data = joints.copy().reshape(len(joints), -1, 3)
    
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    #ax = p3.Axes3D(fig)
    ax = fig.add_subplot(111, projection='3d')
    init()
    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)
    colors_control = ['lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen',
              'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen',
              'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen']
    colors_gen = ['red', 'blue', 'black', 'red', 'blue',
              'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
              'darkred', 'darkred', 'darkred', 'darkred', 'darkred']

    frame_number = data.shape[0]
    #     print(dataset.shape)

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    height_offset_control = data_control.min(axis=0).min(axis=0)[1]
    data_control[:, :, 1] -= height_offset_control
    data_control[..., 0] -= data_control[:, 0:1, 0]
    data_control[..., 2] -= data_control[:, 0:1, 2]

    #     print(trajec.shape)

    def update(index):
        #         print(index)
        ax.cla()

        #ax.lines = []
        #ax.collections = []
        ax.view_init(elev=110, azim=-90) #(120, -90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0, MINS[2] - trajec[index, 1],
                        MAXS[2] - trajec[index, 1])

        if plot_type in ['control_only', 'control_gen']:
            for i, (chain, color) in enumerate(zip(kinematic_tree, colors_control)):
                #             print(color)
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 0.1
                #for j in range(len(data[0])):

                if edit_type in ['inbetweening']:
                    if not np.all(data_control[index, :, 0] == 0):
                        ax.plot3D(data_control[index, chain, 0], data_control[index, chain, 1], data_control[index, chain, 2], marker='o', color=color, linewidth = 0)
                elif edit_type == 'path':
                    ax.plot3D(data_control[index, [0], 0], data_control[index, [0], 1], data_control[index, [0], 2], marker='o', color=color, linewidth = 0)
                elif edit_type == 'upper':
                    ax.plot3D(data_control[index, [0, 2, 5, 8, 11, 1, 4, 7, 10], 0], data_control[index, [0, 2, 5, 8, 11, 1, 4, 7, 10], 1], data_control[index, [0, 2, 5, 8, 11, 1, 4, 7, 10], 2], marker='o', color=color, linewidth = 0)
        if plot_type in ['gen_only', 'control_gen']:    
            for i, (chain, color) in enumerate(zip(kinematic_tree, colors_gen)):
                #             print(color)
                if i < 5:
                    linewidth = 2.0
                else:
                    linewidth = 1.0
                ax.plot3D(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2], linewidth=linewidth,
                        color=color)

        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

    ani = FuncAnimation(fig, update, frames=frame_number,
                        interval=1000 / fps, repeat=False)

    # writer = FFMpegFileWriter(fps=fps)
    ani.save(save_path, fps=fps)
    # ani = FuncAnimation(fig, update, frames=frame_number, interval=1000 / fps, repeat=False, init_func=init)
    # ani.save(save_path, writer='pillow', fps=1000 / fps)

    plt.close()