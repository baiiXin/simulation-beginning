import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import os

class HessianValidator(ABC):
    """
    Hessian验证器基类
    提供通用的顶点导入、碰撞对检测等功能
    """
    
    def __init__(self, system_config: Dict[str, Any] = None):
        self.system_config = system_config or {}
        self.vertices = None
        self.collision_pairs = None
        self.energy_function = None
        
    def load_vertices(self, file_path: str, format_type: str = "obj") -> np.ndarray:
        """
        从文件加载顶点数据
        
        Args:
            file_path: 顶点文件路径
            format_type: 文件格式 ('obj', 'txt', 'npy')
            
        Returns:
            顶点坐标数组
        """
        try:
            if format_type == "obj":
                vertices = self._load_obj_vertices(file_path)
            elif format_type == "txt":
                vertices = np.loadtxt(file_path)
            elif format_type == "npy":
                vertices = np.load(file_path)
            else:
                raise ValueError(f"不支持的格式: {format_type}")
                
            self.vertices = vertices
            print(f"成功加载 {len(vertices)} 个顶点")
            return vertices
            
        except Exception as e:
            print(f"加载顶点失败: {e}")
            return None
    
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
        检测碰撞顶点对
        
        Args:
            distance_threshold: 距离阈值
            
        Returns:
            碰撞对列表 [(i, j), ...]
        """
        if self.vertices is None:
            raise ValueError("请先加载顶点数据")
            
        collision_pairs = []
        n = len(self.vertices)
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = np.linalg.norm(self.vertices[i] - self.vertices[j])
                if distance < distance_threshold:
                    collision_pairs.append((i, j))
        
        self.collision_pairs = collision_pairs
        print(f"检测到 {len(collision_pairs)} 个碰撞对")
        return collision_pairs
    
    def set_energy_function(self, energy_func):
        """设置能量函数"""
        self.energy_function = energy_func
    
    def compute_norm(self, matrix: np.ndarray, norm_type: str = "fro") -> float:
        """
        计算矩阵范数
        
        Args:
            matrix: 输入矩阵
            norm_type: 范数类型 ('fro', 'l1', 'l2', 'inf')
            
        Returns:
            范数值
        """
        if norm_type == "fro":
            return np.linalg.norm(matrix, 'fro')
        elif norm_type == "l1":
            return np.linalg.norm(matrix, 1)
        elif norm_type == "l2":
            return np.linalg.norm(matrix, 2)
        elif norm_type == "inf":
            return np.linalg.norm(matrix, np.inf)
        else:
            raise ValueError(f"不支持的范数类型: {norm_type}")
    
    def relative_error(self, mat1: np.ndarray, mat2: np.ndarray, norm_type: str = "fro") -> float:
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
        return diff_norm / ref_norm if ref_norm > 1e-12 else diff_norm
    
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