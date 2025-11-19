import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np

def plot_pc_3d(pc):
    fig = go.Figure(data=[go.Scatter3d(
        x=pc[:,0], y=pc[:,1], z=pc[:,2],
        mode='markers',
        marker=dict(size=2)
    )])
    fig.update_layout(height=600, width=600)
    fig.show()

def plot_pc_xy(pc):
    plt.figure(figsize=(6,6))
    plt.scatter(pc[:,0], pc[:,1], s=2)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_pc_xz(pc):
    plt.figure(figsize=(6,6))
    plt.scatter(pc[:,0], pc[:,2], s=2)
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_pc_yz(pc):
    plt.figure(figsize=(6,6))
    plt.scatter(pc[:,1], pc[:,2], s=2)
    plt.xlabel("Y")
    plt.ylabel("Z")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

def plot_pc_all_views(pc):
    fig, axs = plt.subplots(1, 3, figsize=(15,5))

    axs[0].scatter(pc[:,0], pc[:,1], s=2)
    axs[0].set_title("XY plane")
    axs[0].set_aspect("equal")

    axs[1].scatter(pc[:,0], pc[:,2], s=2)
    axs[1].set_title("XZ plane")
    axs[1].set_aspect("equal")

    axs[2].scatter(pc[:,1], pc[:,2], s=2)
    axs[2].set_title("YZ plane")
    axs[2].set_aspect("equal")

    plt.show()
