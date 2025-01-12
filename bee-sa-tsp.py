import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

class ABCTSP:
    def __init__(self, n_bees=150, max_trials=200, max_iterations=2000):
        self.n_bees = n_bees
        self.max_trials = max_trials
        self.max_iterations = max_iterations
        self.convergence_count = 0
        self.convergence_threshold = 100  # 收斂閾值
        self.iteration = 0  # 迭代計數器

        # 固定城市座標
        self.cities = np.array([
            [200, 1500], [1000, 2000], [800, 2000], [400, 1500], [600, 2100],  # 0-4
            [600, 2000], [1400, 1500], [1600, 1800], [1000, 1900], [800, 1600],  # 5-9
            [600, 1700], [400, 1700], [1800, 2200], [400, 1500], [200, 1300],  # 10-14
            [600, 1500], [1200, 1500], [200, 1600], [800, 1500], [1000, 1500],  # 15-19
            [600, 2000], [400, 2000], [400, 1600], [1600, 2200], [1200, 1400],  # 20-24
            [400, 2100], [1400, 1600], [1200, 1600], [1000, 2000], [300, 2100]  # 25-29
        ])
        
        self.n_cities = len(self.cities)
        
        # 初始化蜜蜂群
        self.solutions = [self.generate_random_solution() for _ in range(n_bees)]
        self.fitness = [self.calculate_fitness(sol) for sol in self.solutions]
        self.trials = [0] * n_bees
        
        # 初始化最佳解
        self.best_solution = self.generate_random_solution()
        self.best_fitness = self.calculate_fitness(self.best_solution)
        self.best_distance = float('inf')
        
        # 記錄動畫數據
        self.animation_data = []
        self.distance_history = []
        
        # 顏色方案
        self.colors = {
            'cities': '#FF6B6B',      # 城市點顏色
            'path': '#4ECDC4',        # 路徑顏色
            'best_path': '#45B7D1',   # 最佳路徑顏色
            'background': '#F7F7F7'   # 背景顏色
        }

    def generate_random_solution(self):
        """生成隨機解"""
        return list(np.random.permutation(self.n_cities))

    def calculate_distance(self, solution):
        """計算路徑總距離"""
        total_distance = 0
        for i in range(self.n_cities):
            city1 = self.cities[solution[i]]
            city2 = self.cities[solution[(i + 1) % self.n_cities]]
            total_distance += np.sqrt(np.sum((city1 - city2) ** 2))
        return total_distance

    def calculate_fitness(self, solution):
        """計算適應度（距離的倒數）"""
        return 1 / (self.calculate_distance(solution) + 1e-10)

    def employed_bee_phase(self):
        """改進的僱用蜂階段，使用多種局部搜索策略"""
        for i in range(self.n_bees):
            # 使用多種局部搜索策略
            if random.random() < 0.5:
                # 2-opt
                new_solution = self.solutions[i].copy()
                j, k = sorted(random.sample(range(self.n_cities), 2))
                new_solution[j:k+1] = reversed(new_solution[j:k+1])
            else:
                # 3-opt
                new_solution = self.solutions[i].copy()
                points = sorted(random.sample(range(self.n_cities), 3))
                i, j, k = points
                if random.random() < 0.5:
                    # 第一種 3-opt 移動
                    new_solution[i:j], new_solution[j:k] = (
                        new_solution[j:k], new_solution[i:j]
                    )
                else:
                    # 第二種 3-opt 移動
                    segment1 = new_solution[i:j]
                    segment2 = new_solution[j:k]
                    segment3 = new_solution[k:] + new_solution[:i]
                    new_solution = (
                        segment2 + segment1 + segment3
                    )
            
            # 計算新解的適應度
            new_fitness = self.calculate_fitness(new_solution)
            
            # 使用模擬退火準則進行接受
            if new_fitness > self.fitness[i]:
                self.solutions[i] = new_solution
                self.fitness[i] = new_fitness
                self.trials[i] = 0
            else:
                # 以一定概率接受較差的解
                temperature = 1.0 / (self.iteration + 1)
                delta = new_fitness - self.fitness[i]
                if random.random() < np.exp(delta / temperature):
                    self.solutions[i] = new_solution
                    self.fitness[i] = new_fitness
                    self.trials[i] = 0
                else:
                    self.trials[i] += 1

    def onlooker_bee_phase(self):
        """改進的觀察蜂階段，使用輪盤賭選擇和更多的局部搜索"""
        # 計算選擇概率
        total_fitness = sum(self.fitness)
        probabilities = [fit/total_fitness for fit in self.fitness]
        
        for _ in range(self.n_bees):
            # 輪盤賭選擇
            selected = np.random.choice(range(self.n_bees), p=probabilities)
            new_solution = self.solutions[selected].copy()
            
            # 隨機選擇搜索策略
            search_type = random.random()
            
            if search_type < 0.4:  # 40% 概率使用交換
                i, j = random.sample(range(self.n_cities), 2)
                new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            
            elif search_type < 0.7:  # 30% 概率使用插入
                i, j = random.sample(range(self.n_cities), 2)
                if i < j:
                    new_solution = (
                        new_solution[:i] + new_solution[i+1:j+1] +
                        [new_solution[i]] + new_solution[j+1:]
                    )
                else:
                    new_solution = (
                        new_solution[:j] + [new_solution[i]] +
                        new_solution[j:i] + new_solution[i+1:]
                    )
            
            else:  # 30% 概率使用反轉
                i, j = sorted(random.sample(range(self.n_cities), 2))
                new_solution[i:j+1] = reversed(new_solution[i:j+1])
            
            # 計算新解的適應度
            new_fitness = self.calculate_fitness(new_solution)
            
            # 使用貪婪選擇
            if new_fitness > self.fitness[selected]:
                self.solutions[selected] = new_solution
                self.fitness[selected] = new_fitness
                self.trials[selected] = 0
            else:
                self.trials[selected] += 1

    def scout_bee_phase(self):
        """改進的偵查蜂階段，使用貪婪生成初始解"""
        for i in range(self.n_bees):
            if self.trials[i] >= self.max_trials:
                # 使用貪婪策略生成新解
                new_solution = [random.randint(0, self.n_cities-1)]
                unvisited = set(range(self.n_cities)) - {new_solution[0]}
                
                # 貪婪構建路徑
                current = new_solution[0]
                while unvisited:
                    next_city = min(unvisited,
                                  key=lambda x: np.sqrt(np.sum((self.cities[current] -
                                                              self.cities[x])**2)))
                    new_solution.append(next_city)
                    unvisited.remove(next_city)
                    current = next_city
                
                self.solutions[i] = new_solution
                self.fitness[i] = self.calculate_fitness(new_solution)
                self.trials[i] = 0

    def run(self):
        """運行算法"""
        self.iteration = 0  # 重置迭代計數器
        
        while self.iteration < self.max_iterations:
            self.employed_bee_phase()
            self.onlooker_bee_phase()
            self.scout_bee_phase()
            
            # 更新最佳解
            current_best = max(range(self.n_bees), key=lambda i: self.fitness[i])
            current_best_fitness = self.fitness[current_best]
            current_distance = 1/current_best_fitness if current_best_fitness != 0 else float('inf')
            
            # 檢查是否找到更好的解
            if current_best_fitness > self.best_fitness:
                self.best_solution = self.solutions[current_best].copy()
                self.best_fitness = current_best_fitness
                self.best_distance = current_distance
                self.convergence_count = 0
            else:
                self.convergence_count += 1
            
            # 記錄動畫數據
            self.animation_data.append(self.best_solution.copy())
            self.distance_history.append(current_distance)
            
            # 輸出進度
            if self.iteration % 10 == 0:
                print(f"Iteration {self.iteration}/{self.max_iterations}, "
                      f"Best Distance: {self.best_distance:.2f}, "
                      f"Convergence Count: {self.convergence_count}")
            
            # 檢查收斂
            if self.convergence_count >= self.convergence_threshold:
                print(f"\nConverged after {self.iteration} iterations!")
                print(f"Final Best Distance: {self.best_distance:.2f}")
                break
                
            self.iteration += 1

    def visualize_animation(self):
        """創建動畫視覺化"""
        fig = plt.figure(figsize=(16, 9))
        gs = plt.GridSpec(2, 2, width_ratios=[1.5, 1], height_ratios=[2, 1],
                         hspace=0.3, wspace=0.2)

        # 主路徑圖 (左側大圖)
        ax_path = plt.subplot(gs[:, 0])
        
        # 收斂曲線圖 (右上)
        ax_conv = plt.subplot(gs[0, 1])
        
        # 信息面板 (右下)
        ax_info = plt.subplot(gs[1, 1])
        ax_info.axis('off')

        def update(frame):
            # 清除所有軸
            ax_path.clear()
            ax_conv.clear()
            ax_info.clear()
            ax_info.axis('off')

            # 1. 更新主路徑圖
            current_solution = self.animation_data[frame]
            path = np.array([self.cities[current_solution[i]] for i in range(self.n_cities)])
            path = np.vstack([path, path[0]])  # 閉合路徑

            # 設置坐標軸範圍和風格
            ax_path.set_xlim(0, 2000)
            ax_path.set_ylim(1200, 2400)
            ax_path.grid(True, linestyle='--', alpha=0.3)
            ax_path.set_title(f'TSP Path - Iteration {frame + 1}', fontsize=12)

            # 改進路徑繪製
            path_coords = np.array([self.cities[i] for i in current_solution + [current_solution[0]]])
            
            # 使用漸變色繪製路徑
            points = np.array(path_coords)
            segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)
            
            for i, segment in enumerate(segments):
                start, end = segment
                ax_path.plot([start[0], end[0]], [start[1], end[1]], 
                           color='blue', alpha=0.6, linewidth=1.5,
                           zorder=1)
                
                # 添加方向箭頭
                mid_point = (start + end) / 2
                direction = end - start
                arrow_length = np.sqrt(np.sum(direction ** 2))
                if arrow_length > 0:  # 避免除以零
                    direction = direction / arrow_length
                    arrow_size = 20
                    ax_path.arrow(mid_point[0] - direction[0] * arrow_size,
                                mid_point[1] - direction[1] * arrow_size,
                                direction[0] * arrow_size * 2,
                                direction[1] * arrow_size * 2,
                                head_width=15, head_length=20,
                                fc='blue', ec='blue', alpha=0.6,
                                zorder=2)

            # 改進城市點的繪製
            # 一般城市點
            middle_cities = list(range(1, len(self.cities)-1))
            ax_path.scatter(self.cities[middle_cities, 0], self.cities[middle_cities, 1],
                          c='red', s=100, zorder=3, alpha=0.6, marker='o')
            
            # 起點（使用星形）
            start_city = current_solution[0]
            ax_path.scatter(self.cities[start_city, 0], self.cities[start_city, 1],
                          c='green', s=200, zorder=4, alpha=0.8, marker='*',
                          label='Start')
            
            # 終點（使用三角形）
            end_city = current_solution[-1]
            ax_path.scatter(self.cities[end_city, 0], self.cities[end_city, 1],
                          c='purple', s=150, zorder=4, alpha=0.8, marker='^',
                          label='End')
            
            # 添加圖例
            ax_path.legend(loc='upper right', fontsize=10)

            # 改進城市編號標註
            for i, (x, y) in enumerate(self.cities):
                # 根據城市類型選擇顏色和樣式
                if i == current_solution[0]:  # 起點
                    color = 'white'
                    bbox_color = 'green'
                    font_weight = 'bold'
                elif i == current_solution[-1]:  # 終點
                    color = 'white'
                    bbox_color = 'purple'
                    font_weight = 'bold'
                else:  # 中間城市
                    color = 'white'
                    bbox_color = 'red'
                    font_weight = 'normal'
                    
                ax_path.annotate(f'{i}',
                               (x, y),
                               xytext=(0, 0),
                               textcoords='offset points',
                               ha='center',
                               va='center',
                               color=color,
                               fontweight=font_weight,
                               fontsize=8,
                               bbox=dict(facecolor=bbox_color,
                                       alpha=0.6,
                                       edgecolor='none',
                                       pad=1),
                               zorder=4)

            # 2. 更新收斂曲線
            distances = self.distance_history[:frame+1]
            ax_conv.plot(distances, color=self.colors['best_path'], linewidth=2)
            ax_conv.set_title('Convergence History', fontsize=10)
            ax_conv.set_xlabel('Iterations', fontsize=9)
            ax_conv.set_ylabel('Path Length', fontsize=9)
            ax_conv.grid(True, alpha=0.3)
            
            # 添加當前點
            if distances:
                ax_conv.scatter(len(distances)-1, distances[-1], 
                              color='red', s=100, zorder=5)

            # 3. 更新信息面板
            current_distance = self.distance_history[frame]
            historical_best = min(self.distance_history[:frame+1])
            improvement = ((self.distance_history[0] - current_distance) / 
                         self.distance_history[0] * 100)
            
            info_text = [
                f"Iteration: {frame + 1}/{len(self.animation_data)}",
                f"Current Distance: {current_distance:.2f}",
                f"Best Distance: {historical_best:.2f}",
                f"Improvement: {improvement:.1f}%"
            ]
            
            # 在右下角面板顯示信息
            y_pos = 0.95
            for i, line in enumerate(info_text):
                ax_info.text(0.05, y_pos, line,
                           fontsize=11,
                           fontweight='bold' if i == 0 else 'normal',
                           transform=ax_info.transAxes)
                y_pos -= 0.2

            plt.tight_layout()

            # 設置統一的背景顏色
            fig.patch.set_facecolor(self.colors['background'])
            ax_path.set_facecolor(self.colors['background'])
            ax_conv.set_facecolor(self.colors['background'])
            ax_info.set_facecolor(self.colors['background'])

        anim = FuncAnimation(fig, update,
                           frames=len(self.animation_data),
                           interval=100,
                           repeat=False)
        plt.show()

if __name__ == "__main__":
    # 設置隨機種子
    np.random.seed(42)
    random.seed(42)
    
    # 創建並運行算法
    abc = ABCTSP(
        n_bees=150,          # 蜜蜂數量
        max_trials=200,      # 最大試驗次數
        max_iterations=2000  # 最大迭代次數
    )
    
    # 運行算法
    abc.run()
    
    # 顯示動畫
    abc.visualize_animation()