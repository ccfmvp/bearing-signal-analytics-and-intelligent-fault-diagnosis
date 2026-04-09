import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import re
from pathlib import Path


class ModelVisualizer3D:
    def __init__(self, model_path, output_dir="model_visualizations"):
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.model_data = None

        # 创建输出目录
        self.output_dir.mkdir(exist_ok=True)

        # 设置美观的样式
        plt.style.use('seaborn-v0_8')

    def load_model(self):
        """加载.pth模型文件"""
        try:
            self.model_data = torch.load(self.model_path, map_location='cpu')
            print("模型加载成功!")
            return True
        except Exception as e:
            print(f"加载模型失败: {e}")
            return False

    def sanitize_filename(self, name):
        """清理层名称以创建有效的文件名"""
        # 替换非法字符
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        # 限制长度
        if len(name) > 50:
            name = name[:50]
        return name

    def create_3d_surface(self, tensor, layer_name, param_type):
        """创建3D表面图"""
        if tensor.dim() < 2:
            print(f"张量维度不足2D，跳过 {layer_name}.{param_type}")
            return

        # 如果是高维张量，取前两个维度
        if tensor.dim() > 2:
            tensor = tensor.view(tensor.size(0), -1)

        # 确保是2D矩阵
        if tensor.dim() == 2:
            data = tensor.detach().numpy()

            fig = plt.figure(figsize=(16, 12))
            ax = fig.add_subplot(111, projection='3d')

            # 创建网格
            x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

            # 绘制表面图
            surf = ax.plot_surface(x, y, data, cmap='viridis', alpha=0.8,
                                   linewidth=0, antialiased=True)

            # 设置标题和标签
            ax.set_title(f'{layer_name} - {param_type}\nShape: {tensor.shape}',
                         fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('X Dimension', fontsize=12, labelpad=10)
            ax.set_ylabel('Y Dimension', fontsize=12, labelpad=10)
            ax.set_zlabel('Parameter Value', fontsize=12, labelpad=10)

            # 添加颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)

            # 设置视角
            ax.view_init(elev=30, azim=45)

            # 保存图片
            filename = f"{param_type}_surface.png"
            plt.savefig(self.output_dir / layer_name / filename,
                        dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()

    def create_3d_scatter(self, tensor, layer_name, param_type):
        """创建3D散点图"""
        if tensor.numel() < 3:
            print(f"元素数量不足，跳过 {layer_name}.{param_type}")
            return

        # 展平张量
        flat_tensor = tensor.flatten().detach().numpy()

        # 如果元素太多，进行采样以避免过载
        max_points = 10000
        if len(flat_tensor) > max_points:
            indices = np.random.choice(len(flat_tensor), max_points, replace=False)
            flat_tensor = flat_tensor[indices]

        # 创建3D坐标
        x = np.arange(len(flat_tensor))
        y = np.zeros_like(x)
        z = flat_tensor

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 创建散点图，颜色基于值
        scatter = ax.scatter(x, y, z, c=z, cmap='plasma', alpha=0.7,
                             s=20, linewidth=0.5, edgecolor='white')

        ax.set_title(f'{layer_name} - {param_type}\nShape: {tensor.shape}',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Parameter Index', fontsize=12, labelpad=10)
        ax.set_ylabel('Y', fontsize=12, labelpad=10)
        ax.set_zlabel('Parameter Value', fontsize=12, labelpad=10)

        fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        ax.view_init(elev=20, azim=45)

        filename = f"{param_type}_scatter.png"
        plt.savefig(self.output_dir / layer_name / filename,
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_3d_histogram(self, tensor, layer_name, param_type):
        """创建3D直方图"""
        data = tensor.flatten().detach().numpy()

        if len(data) == 0:
            return

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        # 创建直方图
        hist, bins = np.histogram(data, bins=50)
        x = (bins[:-1] + bins[1:]) / 2
        y = np.zeros_like(x)

        # 绘制3D条形图
        dx = (bins[1] - bins[0]) * 0.8
        dy = dx
        ax.bar3d(x, y, np.zeros_like(hist), dx, dy, hist,
                 color='skyblue', alpha=0.8, edgecolor='black', linewidth=0.5)

        ax.set_title(f'{layer_name} - {param_type}\nDistribution',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Parameter Value', fontsize=12, labelpad=10)
        ax.set_ylabel('Y', fontsize=12, labelpad=10)
        ax.set_zlabel('Frequency', fontsize=12, labelpad=10)

        ax.view_init(elev=25, azim=45)

        filename = f"{param_type}_histogram.png"
        plt.savefig(self.output_dir / layer_name / filename,
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def create_heatmap_3d(self, tensor, layer_name, param_type):
        """创建3D热力图"""
        if tensor.dim() < 2:
            return

        if tensor.dim() > 2:
            tensor = tensor.view(tensor.size(0), -1)

        data = tensor.detach().numpy()

        fig = plt.figure(figsize=(16, 12))
        ax = fig.add_subplot(111, projection='3d')

        x, y = np.meshgrid(np.arange(data.shape[1]), np.arange(data.shape[0]))

        # 创建3D热力图
        surf = ax.plot_surface(x, y, data, cmap='hot', alpha=0.9,
                               linewidth=0, antialiased=True)

        ax.set_title(f'{layer_name} - {param_type}\nHeatmap',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('X Dimension', fontsize=12, labelpad=10)
        ax.set_ylabel('Y Dimension', fontsize=12, labelpad=10)
        ax.set_zlabel('Parameter Value', fontsize=12, labelpad=10)

        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, pad=0.1)
        ax.view_init(elev=30, azim=45)

        filename = f"{param_type}_heatmap.png"
        plt.savefig(self.output_dir / layer_name / filename,
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

    def visualize_layer(self, layer_name, tensor, param_type):
        """为单个层的参数创建多种3D可视化"""
        # 创建层子目录
        layer_dir = self.output_dir / self.sanitize_filename(layer_name)
        layer_dir.mkdir(exist_ok=True)

        print(f"可视化层: {layer_name}.{param_type}, 形状: {tensor.shape}")

        # 根据张量维度选择合适的可视化方法
        try:
            if tensor.dim() >= 2:
                self.create_3d_surface(tensor, layer_name, param_type)
                self.create_heatmap_3d(tensor, layer_name, param_type)

            if tensor.numel() > 10:  # 只有足够多的元素时才创建散点图和直方图
                self.create_3d_scatter(tensor, layer_name, param_type)
                self.create_3d_histogram(tensor, layer_name, param_type)

        except Exception as e:
            print(f"可视化 {layer_name}.{param_type} 时出错: {e}")

    def extract_parameters(self, obj, prefix=""):
        """递归提取模型参数"""
        parameters = {}

        if isinstance(obj, torch.Tensor):
            parameters[prefix] = obj
        elif isinstance(obj, dict):
            for key, value in obj.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                parameters.update(self.extract_parameters(value, new_prefix))
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                new_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
                parameters.update(self.extract_parameters(item, new_prefix))
        elif hasattr(obj, 'state_dict'):  # 如果是模型对象
            parameters.update(self.extract_parameters(obj.state_dict(), prefix))
        elif hasattr(obj, '__dict__'):  # 其他Python对象
            parameters.update(self.extract_parameters(obj.__dict__, prefix))

        return parameters

    def visualize_all_layers(self):
        """可视化所有层"""
        if not self.load_model():
            return

        # 提取所有参数
        parameters = self.extract_parameters(self.model_data)

        if not parameters:
            print("未找到可可视化的参数")
            return

        print(f"找到 {len(parameters)} 个参数组")

        # 为每个参数创建可视化
        for layer_name, tensor in parameters.items():
            if tensor.dim() == 0:  # 跳过标量
                continue

            param_type = "weight" if "weight" in layer_name.lower() else "bias" if "bias" in layer_name.lower() else "param"
            self.visualize_layer(layer_name, tensor, param_type)

        print(f"\n所有可视化图已保存到: {self.output_dir.absolute()}")

        # 生成汇总报告
        self.generate_summary_report(parameters)

    def generate_summary_report(self, parameters):
        """生成汇总报告"""
        report_path = self.output_dir / "visualization_summary.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("模型参数可视化汇总报告\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"模型文件: {self.model_path}\n")
            f.write(f"总参数组数: {len(parameters)}\n")
            f.write(f"输出目录: {self.output_dir.absolute()}\n\n")

            f.write("各层参数详情:\n")
            f.write("-" * 50 + "\n")

            total_params = 0
            for layer_name, tensor in parameters.items():
                param_count = tensor.numel()
                total_params += param_count
                f.write(f"{layer_name:<40} | 形状: {str(tensor.shape):<20} | 参数数量: {param_count:>10,}\n")

            f.write(f"\n总参数数量: {total_params:,}\n")

        print(f"汇总报告已保存到: {report_path}")


def main():
    # 使用示例
    model_path = "../第四小问代码/问题四模型/resnet_transformer_mmd.pth"  # 替换为你的.pth文件路径
    output_dir = "../第四小问代码/问题四模型/model_3d_visualizations/resnet_transformer_mmd"

    # 创建可视化器
    visualizer = ModelVisualizer3D(model_path, output_dir)

    # 开始可视化
    visualizer.visualize_all_layers()


if __name__ == "__main__":
    main()