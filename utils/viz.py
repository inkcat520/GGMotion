import os

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

gt_color = ["#8e8e8e", "#383838"]
pre_color = ["#F48ECF", "#3535D1"]

class Ax3DPose:
    def __init__(self, ax, label=('GT', 'Pred')):
        """
        Create a 3d pose visualizer that can be updated with new poses.

        Args
          ax: 3d axis to plot the 3d pose on
          lcolor: String. Colour for the left part of the body
          rcolor: String. Colour for the right part of the body
        """

        # Start and endpoints of our representation
        self.I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 15, 14, 18, 19, 14, 26, 27]) - 1
        self.J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 15, 16, 18, 19, 20, 26, 27, 28]) - 1
        # Left / right indicator
        self.LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=bool)
        self.ax = ax

        vals = np.zeros((32, 3))

        # Make connection matrix
        self.plots = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots.append(
                    self.ax.plot(x, z, y, lw=2, linestyle='--', c='#FFFFFF', label=label[0]))
            else:
                self.plots.append(self.ax.plot(x, y, z, lw=2, linestyle='--', c='#FFFFFF'))

        self.plots_pred = []
        for i in np.arange(len(self.I)):
            x = np.array([vals[self.I[i], 0], vals[self.J[i], 0]])
            y = np.array([vals[self.I[i], 1], vals[self.J[i], 1]])
            z = np.array([vals[self.I[i], 2], vals[self.J[i], 2]])
            if i == 0:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='#FFFFFF', label=label[1]))
            else:
                self.plots_pred.append(self.ax.plot(x, y, z, lw=2, c='#FFFFFF'))

        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.set_zlabel("z")
        # self.ax.set_axis_off()
        # self.ax.axes.get_xaxis().set_visible(False)
        # self.axes.get_yaxis().set_visible(False)
        self.ax.view_init(120, -90)

    def update(self, gt_channels, pred_channels, color=("#F48ECF", "#3535D1")):
        """
        Update the plotted 3d pose.

        Args
          channels: 96-dim long np array. The pose to plot.
          lcolor: String. Colour for the left part of the body.
          rcolor: String. Colour for the right part of the body.
        Returns
          Nothing. Simply updates the axis with the new pose.
        """

        assert gt_channels.size == 96, "channels should have 96 entries, it has %d instead" % gt_channels.size
        gt_vals = np.reshape(gt_channels, (32, -1))
        lcolor = "#8e8e8e"
        rcolor = "#383838"
        for i in np.arange(len(self.I)):
            x = np.array([gt_vals[self.I[i], 0], gt_vals[self.J[i], 0]])
            y = np.array([gt_vals[self.I[i], 1], gt_vals[self.J[i], 1]])
            z = np.array([gt_vals[self.I[i], 2], gt_vals[self.J[i], 2]])
            self.plots[i][0].set_xdata(x)
            self.plots[i][0].set_ydata(y)
            self.plots[i][0].set_3d_properties(z)
            self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots[i][0].set_alpha(0.5)

        assert pred_channels.size == 96, "channels should have 96 entries, it has %d instead" % pred_channels.size
        pred_vals = np.reshape(pred_channels, (32, -1))
        lcolor = color[0]
        rcolor = color[1]
        for i in np.arange(len(self.I)):
            x = np.array([pred_vals[self.I[i], 0], pred_vals[self.J[i], 0]])
            y = np.array([pred_vals[self.I[i], 1], pred_vals[self.J[i], 1]])
            z = np.array([pred_vals[self.I[i], 2], pred_vals[self.J[i], 2]])
            self.plots_pred[i][0].set_xdata(x)
            self.plots_pred[i][0].set_ydata(y)
            self.plots_pred[i][0].set_3d_properties(z)
            self.plots_pred[i][0].set_color(lcolor if self.LR[i] else rcolor)
            # self.plots_pred[i][0].set_alpha(0.7)

        r = 750
        xroot, yroot, zroot = gt_vals[0, 0], gt_vals[0, 1], gt_vals[0, 2]
        self.ax.set_xlim3d([-r + xroot, r + xroot])
        self.ax.set_zlim3d([-r + zroot, r + zroot])
        self.ax.set_ylim3d([-r + yroot, r + yroot])
        self.ax.set_aspect('auto')
        self.ax.legend(loc='lower left', bbox_to_anchor=(1, 1))


def plot_predictions(gt, pred, gt_pre, f_title, save_path, frame_select=None, duration=0.1):
    # 创建一个窗口和子图
    plt.close('all')
    fig = plt.figure(figsize=(8, 6), dpi=120)
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(left=0, right=1, top=0.85, bottom=0)
    crop = [245, -100, 390, -360]

    nframes_pred = pred.shape[0]

    # 创建一个3D姿势对象
    ob = Ax3DPose(ax)

    image_list = []
    # 显示窗口，但不要阻塞程序
    plt.show(block=False)
    title_text = ax.set_title("", loc='center', pad=30.0, color="#383838")
    # 循环显示每一帧
    for i in range(nframes_pred):
        # 更新姿势
        ob.update(gt[i, :], pred[i, :], pre_color)

        # 更新标题文字
        title_text.set_text(f_title + ' frame:{:d}'.format(i + 1))
        # ax.axis('off')  # 隐藏坐标轴
        # ax.legend().set_visible(False)  # 隐藏图例

        # 强制重新绘制
        fig.canvas.draw()
        image_list.append(np.array(fig.canvas.renderer.buffer_rgba()))

        # ax.axis('on')
        # ax.legend().set_visible(True)
        # 等待一段时间，以便有足够的时间观察图形变化
        plt.pause(duration)

    # 关闭窗口
    imageio.mimsave(save_path + ".gif", image_list, duration=duration)

