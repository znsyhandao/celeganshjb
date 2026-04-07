import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class HJBController:
    def __init__(self, lambda_energy=0.05, sigma_noise=0.1, goal=70.0):
        self.lambda_energy = lambda_energy
        self.sigma_noise = sigma_noise
        self.goal = goal
        
        # 使用更大的状态代价
        q = 2.0  # 状态代价系数
        
        # Riccati方程
        p = np.sqrt(q * self.lambda_energy / 2)
        self.K = 2 * p / self.lambda_energy
        
        # 直接使用线性控制（不限幅，让自然饱和）
        print(f"\n控制器参数:")
        print(f"  lambda = {self.lambda_energy}")
        print(f"  sigma = {self.sigma_noise}")
        print(f"  K = {self.K:.4f}")
        print(f"  控制律: u = -{self.K:.4f} * (x - {goal})")
        
    def control(self, x):
        """直接线性控制，不添加非线性"""
        error = x - self.goal
        u = -self.K * error
        # 软限幅：允许超过1，但会自然饱和
        return np.clip(u, -1.5, 1.5)
    
    def simulate_trajectory(self, x0, T=30.0, dt=0.05):
        """模拟单条轨迹"""
        n_steps = int(T / dt)
        trajectory = np.zeros(n_steps + 1)
        trajectory[0] = x0
        
        for i in range(n_steps):
            x = trajectory[i]
            u = self.control(x)
            
            # 减少噪声，增加确定性控制
            u_noisy = u + np.random.normal(0, 0.02)
            
            # SDE更新
            dW = np.random.normal(0, np.sqrt(dt))
            dx = u_noisy * dt + self.sigma_noise * dW
            x_new = x + dx
            
            # 简单边界
            x_new = max(0, min(100, x_new))
            
            trajectory[i+1] = x_new
        
        return trajectory


def main():
    print("=" * 70)
    print("HJB随机最优控制 - 线虫趋化模型")
    print("=" * 70)
    
    # 最佳参数
    lambda_energy = 0.05
    sigma_noise = 0.08
    controller = HJBController(lambda_energy, sigma_noise, goal=70.0)
    
    # 模拟
    n_trials = 100
    print(f"\n运行 {n_trials} 次模拟...")
    
    trajectories = []
    final_positions = []
    
    for i in range(n_trials):
        traj = controller.simulate_trajectory(20.0, T=30.0, dt=0.05)
        trajectories.append(traj)
        final_positions.append(traj[-1])
        
        if (i+1) % 20 == 0:
            print(f"  完成 {i+1}/{n_trials}")
    
    # 统计
    success = np.mean([abs(p-70) < 15 for p in final_positions]) * 100
    mean_final = np.mean(final_positions)
    std_final = np.std(final_positions)
    
    print(f"\n结果:")
    print(f"  平均最终位置: {mean_final:.1f} +/- {std_final:.1f}")
    print(f"  成功率 (|x-70|<15): {success:.1f}%")
    
    # 绘图
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # 控制律
    ax1 = axes[0]
    x_plot = np.linspace(0, 100, 200)
    u_plot = [controller.control(x) for x in x_plot]
    ax1.plot(x_plot, u_plot, 'b-', linewidth=2)
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax1.axvline(70, color='r', linestyle='--', label='Goal')
    ax1.axvline(20, color='g', linestyle='--', label='Start')
    ax1.set_xlabel('Position'); ax1.set_ylabel('Control u')
    ax1.set_title('Optimal Control Policy')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    
    # 轨迹
    ax2 = axes[1]
    for traj in trajectories[:30]:
        times = np.linspace(0, 30, len(traj))
        ax2.plot(times, traj, 'b-', alpha=0.3, linewidth=0.8)
    mean_traj = np.mean(trajectories, axis=0)
    ax2.plot(times, mean_traj, 'r-', linewidth=2, label='Mean')
    ax2.axhline(70, color='g', linestyle='--', label='Goal')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Position')
    ax2.set_title('Trajectories')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    
    # 最终位置分布
    ax3 = axes[2]
    ax3.hist(final_positions, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax3.axvline(70, color='g', linestyle='--', label='Goal')
    ax3.axvline(mean_final, color='r', linestyle='--', label=f'Mean={mean_final:.1f}')
    ax3.set_xlabel('Final Position'); ax3.set_ylabel('Frequency')
    ax3.set_title(f'Final Distribution (Success: {success:.0f}%)')
    ax3.legend(); ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return success


if __name__ == "__main__":
    success = main()
    
    print("\n" + "=" * 70)
    print("数学原理")
    print("=" * 70)
    print("""
HJB方程: 0 = min_u [c(x) + λ·r(u) + V_x·u + ½σ²V_xx]

对于二次代价 c(x) = q·(x-goal)²:
  - 值函数: V(x) = p·(x-goal)²
  - Riccati方程: 2p²/λ = q → p = √(qλ/2)
  - 最优控制: u* = - (2p/λ)·(x-goal) = -K·(x-goal)

随机微分方程: dx = u* dt + σ dW
    """)
    
    if success > 30:
        print("✅ 成功验证HJB最优控制理论")
    else:
        print("⚠️ 请手动调整参数: lambda=0.03-0.08, sigma=0.05-0.1")