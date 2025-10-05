import torch
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import os

class HessianValidator(ABC):
    """
    Hessian验证器基类 - PyTorch版本
    提供通用的顶点导入、碰撞对检测等功能
    """
    
    def __init__(self, system_config: Dict[str, Any] = None, device: str = None):
        self.system_config = system_config or {}
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.vertices = None
        self.collision_pairs = None
        self.energy_function = None
        
    def load_vertices(self, vertices_data: Any = None, file_path: str = None, format_type: str = "obj") -> torch.Tensor:
        """
        加载顶点数据到PyTorch张量
        支持直接传入numpy数组或从文件加载
        
        Args:
            vertices_data: 直接传入的顶点数据 (numpy数组、PyTorch张量、列表等)
            file_path: 顶点文件路径（与vertices_data二选一）
            format_type: 文件格式 ('obj', 'txt', 'npy')
            
        Returns:
            顶点坐标张量
        """
        try:
            # 情况1: 直接传入数据
            if vertices_data is not None:
                vertices = self._process_direct_input(vertices_data)
            
            # 情况2: 从文件加载
            elif file_path is not None:
                vertices = self._load_from_file(file_path, format_type)
            
            else:
                raise ValueError("必须提供vertices_data或file_path参数")
            
            # 转换为PyTorch张量并移到设备
            self.vertices = torch.tensor(vertices, dtype=torch.float64, device=self.device)
            print(f"成功加载 {len(vertices)} 个顶点到 {self.device}")
            return self.vertices
        
        except Exception as e:
            print(f"加载顶点失败: {e}")
            return None

    def _process_direct_input(self, vertices_data: Any) -> np.ndarray:
        """
        处理直接传入的顶点数据
        """
        # 如果已经是numpy数组，直接返回
        if isinstance(vertices_data, np.ndarray):
            return vertices_data
        
        # 如果是PyTorch张量，转换为numpy
        elif isinstance(vertices_data, torch.Tensor):
            return vertices_data.cpu().numpy()
        
        # 如果是Python列表，转换为numpy数组
        elif isinstance(vertices_data, (list, tuple)):
            return np.array(vertices_data)
        
        # 如果是其他可迭代对象
        elif hasattr(vertices_data, '__array__'):
            return np.array(vertices_data)
        
        else:
            raise ValueError(f"不支持的顶点数据类型: {type(vertices_data)}")

    def _load_from_file(self, file_path: str, format_type: str) -> np.ndarray:
        """
        从文件加载顶点数据
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        if format_type == "obj":
            return self._load_obj_vertices(file_path)
        elif format_type == "txt":
            return np.loadtxt(file_path)
        elif format_type == "npy":
            return np.load(file_path)
        else:
            raise ValueError(f"不支持的格式: {format_type}")

    def _load_obj_vertices(self, file_path: str) -> np.ndarray:
        """加载OBJ格式的顶点数据"""
        vertices = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    vertex = list(map(float, line.strip().split()[1:4]))
                    vertices.append(vertex)
        return np.array(vertices)
    
    def detect_collision_pairs(self, distance_threshold: float = 0.1) -> List[Tuple[int, int]]:
        """
        检测碰撞顶点对 - 使用PyTorch计算
        
        Args:
            distance_threshold: 距离阈值
            
        Returns:
            碰撞对列表 [(i, j), ...]
        """
        if self.vertices is None:
            raise ValueError("请先加载顶点数据")
            
        collision_pairs = []
        n = len(self.vertices)
        
        # 使用PyTorch批量计算距离
        for i in range(n):
            # 计算顶点i到所有其他顶点的距离
            distances = torch.norm(self.vertices - self.vertices[i], dim=1)
            # 找到距离小于阈值的顶点（排除自身）
            close_indices = torch.where((distances < distance_threshold) & (torch.arange(n, device=self.device) != i))[0]
            
            for j in close_indices:
                if i < j.item():  # 避免重复
                    collision_pairs.append((i, j.item()))
        
        self.collision_pairs = collision_pairs
        print(f"检测到 {len(collision_pairs)} 个碰撞对")
        return collision_pairs
    
    def set_energy_function(self, energy_func):
        """设置能量函数"""
        self.energy_function = energy_func
    
    def compute_norm(self, matrix: torch.Tensor, norm_type: str = "fro") -> float:
        """
        计算矩阵范数 - PyTorch版本
        
        Args:
            matrix: 输入矩阵
            norm_type: 范数类型 ('fro', 'l1', 'l2', 'inf')
            
        Returns:
            范数值
        """
        if norm_type == "fro":
            return torch.norm(matrix, p='fro').item()
        elif norm_type == "l1":
            return torch.norm(matrix, p=1).item()
        elif norm_type == "l2":
            return torch.norm(matrix, p=2).item()
        elif norm_type == "inf":
            return torch.norm(matrix, p=float('inf')).item()
        else:
            raise ValueError(f"不支持的范数类型: {norm_type}")
    
    def relative_error(self, mat1: torch.Tensor, mat2: torch.Tensor, norm_type: str = "inf") -> float:
        """
        计算两个矩阵的相对误差
        
        Args:
            mat1: 矩阵1
            mat2: 矩阵2
            norm_type: 范数类型
            
        Returns:
            相对误差
        """
        diff_norm = self.compute_norm(mat1 - mat2, norm_type)
        ref_norm = self.compute_norm(mat1, norm_type)
        return diff_norm
    
    def to_device(self, data: Any) -> Any:
        """将数据移动到设备"""
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return type(data)(self.to_device(x) for x in data)
        elif isinstance(data, dict):
            return {k: self.to_device(v) for k, v in data.items()}
        else:
            return data
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> Dict[str, Any]:
        """验证方法，子类必须实现"""
        pass
    
    def print_validation_summary(self, results: Dict[str, Any]):
        """打印验证结果摘要"""
        print("\n" + "="*50)
        print("Hessian验证结果摘要")
        print("="*50)
        for key, value in results.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6e}")
            else:
                print(f"{key}: {value}")
        print("="*50)