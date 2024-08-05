import os
import numpy as np
from ale_py import ALEInterface
from cluster_tool import ClusterTool


# 初始化Atari环境
def init_atari_env(rom_path, seed=0):
    if not os.path.exists(rom_path):
        raise FileNotFoundError(f"ROM file '{rom_path}' not found.")

    ale = ALEInterface()
    ale.setInt("random_seed", seed)
    ale.loadROM(rom_path)
    return ale


# 收集状态样本
def collect_state_samples(ale, num_samples=1000):
    states = []
    ale.reset_game()
    for _ in range(num_samples):
        action = np.random.randint(len(ale.getLegalActionSet()))  # 随机动作
        ale.act(action)
        state = ale.getScreenRGB()
        states.append(state)
        if ale.game_over():
            ale.reset_game()
    return np.array(states)


# 主函数
def main():
    rom_path = "/Users/haydengu/Documents/Research Project/cognitive-belief-driven-qlearning/Roms/Breakout - Breakaway IV.bin"  # 需要将此文件放置在当前目录或指定路径
    num_samples = 1000  # 样本数量
    n_clusters = 10  # 聚类数量

    try:
        # 初始化环境
        ale = init_atari_env(rom_path)
        print("Environment initialized.")
    except FileNotFoundError as e:
        print(e)
        return

    # 收集状态样本
    state_samples = collect_state_samples(ale, num_samples)
    print("State samples collected:", state_samples.shape)

    # 获取状态空间的形状并进行必要的预处理
    state_samples = state_samples.reshape(num_samples, -1)

    # 初始化ClusterTool并进行聚类
    cluster_tool = ClusterTool(state_space=state_samples, action_space=len(ale.getLegalActionSet()),
                               n_clusters=n_clusters)
    print("Clustering completed.")

    # 打印每个簇的样本数量
    cluster_counts = np.bincount(cluster_tool.state_clusters)
    for i, count in enumerate(cluster_counts):
        print(f"Cluster {i}: {count} samples")


if __name__ == "__main__":
    main()
