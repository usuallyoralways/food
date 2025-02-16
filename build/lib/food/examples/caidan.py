import tkinter as tk
import random
import math
 
# 心形参数
heartx, hearty = 200, 200  # 心形中心坐标
side = 10  # 心形大小参数
heartcolor = "red"  # 心形颜色
 
 
class HeartAnimation:
    def __init__(self, generate_frame=20):
        self._points = set()  # 原始爱心坐标集合
        self._edge_diffusion_points = set()  # 边缘扩散效果点坐标集合
        self._center_diffusion_points = set()  # 中心扩散效果点坐标集合
        self.all_points = {}  # 每帧动态点坐标
        self.build(2000)  # 构建初始爱心点
        self.random_halo = 1000  # 随机光环效果的参数
        self.generate_frame = generate_frame  # 生成的帧数
        for frame in range(generate_frame):
            self.calc(frame)  # 计算每帧的点
 
    def build(self, number):
        # 随机生成心形的原始点
        for _ in range(number):
            t = random.uniform(0, 2 * math.pi)  # 随机角度
            x, y = self.heart_function(t)  # 计算心形坐标
            self._points.add((x, y))  # 添加到原始坐标集合
 
        # 处理边缘扩散点
        for _x, _y in list(self._points):
            for _ in range(3):
                x, y = self.scatter_inside(_x, _y, 0.05)  # 生成散落点
                self._edge_diffusion_points.add((x, y))  # 添加到边缘扩散集合
 
        # 处理中心扩散点
        point_list = list(self._points)
        for _ in range(4000):
            x, y = random.choice(point_list)  # 随机选择心形中的一个点
            x, y = self.scatter_inside(x, y, 0.17)  # 生成中心散落点
            self._center_diffusion_points.add((x, y))  # 添加到中心扩散集合
 
    @staticmethod
    def calc_position(x, y, ratio):
        # 计算点的扩散位置
        force = 1 / (((x - heartx) ** 2 + (y - hearty) ** 2) ** 0.520)  # 魔法参数
        dx = ratio * force * (x - heartx) + random.randint(-1, 1)  # 计算x方向扩散
        dy = ratio * force * (y - hearty) + random.randint(-1, 1)  # 计算y方向扩散
        return x + dx, y + dy  # 返回新的扩散坐标
 
    def calc(self, generate_frame):
        # 计算当前帧的所有点
        ratio = 10 * self.curve(generate_frame / 10 * math.pi)  # 圆滑的周期缩放比例
        halo_radius = int(4 + 6 * (1 + self.curve(generate_frame / 10 * math.pi)))  # 光环半径
        halo_number = int(3000 + 4000 * abs(self.curve(generate_frame / 10 * math.pi) ** 2))  # 光环数量
        all_points = []  # 所有点的集合
        heart_halo_point = set()  # 存储光环点集合
 
        # 创建扩散粒子
        for _ in range(halo_number):
            t = random.uniform(0, 2 * math.pi)  # 随机角度
            x, y = self.heart_function(t, shrink_ratio=11.6)  # 计算光环点坐标
            x, y = self.shrink(x, y, halo_radius)  # 收缩点位置
            if (x, y) not in heart_halo_point:
                heart_halo_point.add((x, y))  # 确保光环点唯一
                x += random.randint(-14, 14)  # 随机位移
                y += random.randint(-14, 14)  # 随机位移
                size = random.choice((1, 2, 2))  # 随机大小
                all_points.append((x, y, size))  # 添加到所有点集合
 
        # 处理心形轮廓点
        for x, y in self._points:
            x, y = self.calc_position(x, y, ratio)  # 计算扩散位置
            size = random.randint(1, 3)  # 随机大小
            all_points.append((x, y, size))  # 添加到所有点集合
 
        # 处理边缘扩散点
        for x, y in self._edge_diffusion_points:
            x, y = self.calc_position(x, y, ratio)  # 计算扩散位置
            size = random.randint(1, 2)  # 随机大小
            all_points.append((x, y, size))  # 添加到所有点集合
 
        # 处理中心扩散点
        for x, y in self._center_diffusion_points:
            x, y = self.calc_position(x, y, ratio)  # 计算扩散位置
            size = random.randint(1, 2)  # 随机大小
            all_points.append((x, y, size))  # 添加到所有点集合
 
        self.all_points[generate_frame] = all_points  # 保存当前帧所有点
 
    def render(self, render_canvas, render_frame):
        # 在画布上绘制当前帧的点
        for x, y, size in self.all_points[render_frame % self.generate_frame]:
            render_canvas.create_rectangle(x, y, x + size, y + size, width=0, fill=heartcolor)  # 绘制矩形
 
    @staticmethod
    def heart_function(t, shrink_ratio: float = side):
        # 计算心形的坐标
        x = 16 * (math.sin(t) ** 3)  # x坐标公式
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))  # y坐标公式
        x *= shrink_ratio  # 应用缩放
        y *= shrink_ratio  # 应用缩放
        x += heartx  # 移动到中心
        y += hearty  # 移动到中心
        return int(x), int(y)  # 返回整数坐标
 
    @staticmethod
    def scatter_inside(x, y, beta=0.15):
        # 生成心形内部的随机散落点
        ratio_x = -beta * math.log(random.random())  # 计算x方向散落量
        ratio_y = -beta * math.log(random.random())  # 计算y方向散落量
        dx = ratio_x * (x - heartx)  # 根据心形中心计算x偏移
        dy = ratio_y * (y - hearty)  # 根据心形中心计算y偏移
        return x - dx, y - dy  # 返回新的坐标
 
    @staticmethod
    def shrink(x, y, ratio):
        # 计算点的收缩位置
        force = -1 / (((x - heartx) ** 2 + (y - hearty) ** 2) ** 0.6)
        dx = ratio * force * (x - heartx)  # 根据心形中心计算x偏移
        dy = ratio * force * (y - hearty)  # 根据心形中心计算y偏移
        return x - dx, y - dy  # 返回新的坐标
 
    @staticmethod
    def curve(p):
        # 计算曲线函数
        return 2 * (2 * math.sin(4 * p)) / (2 * math.pi)
 
 
def draw(main: tk.Tk, render_canvas: tk.Canvas, render_heart: HeartAnimation, render_frame=0):
    # 绘制函数
    render_canvas.delete('all')  # 清空画布
    render_heart.render(render_canvas, render_frame)  # 绘制当前帧
    main.after(160, draw, main, render_canvas, render_heart, render_frame + 1)  # 定时调用下一帧
 
 
def caidan():
    root = tk.Tk()  # 创建主窗口
    root.configure(bg='black')  # 设置窗口背景为黑色
    root.geometry("400x400")  # 设置窗口大小
    root.title('love chen') 
 
    canvas = tk.Canvas(root, width=400, height=400, bg='black')  # 创建画布
    canvas.pack(fill=tk.BOTH, expand=True)  # 自适应大小
    heart_animation = HeartAnimation(generate_frame=60)  # 创建心形动画对象
    draw(root, canvas, heart_animation)  # 启动绘制
    root.mainloop()  # 运行主循环



if __name__ == "__main__":
    caidan()