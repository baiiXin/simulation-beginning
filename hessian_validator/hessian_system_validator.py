import numpy as np
from typing import Dict, Any, Callable, List
from hessian_validator import HessianValidator
from hessian_point_validator import HessianPointValidator

class HessianSystemValidator(HessianValidator):
    """
    系统级Hessian验证器
    验证整个系统的Hessian矩阵一致性
    """
    
    def __init__(self, system_config: Dict[str, Any] = None):
        super().__init__(system_config)
        self.point_validator = HessianPointValidator(system_config)
        self.sampling_strategies = ['random', 'uniform_grid', 'critical_points']
        
    def validate(self,
                system_energy_func: Callable = None,
                sampling_strategy: str = 'random',
                num_samples: int = 10,
                tolerance: float = 1e-4) -> Dict[str, Any]:
        """
        系统级Hessian验证
        
        Args:
            system_energy_func: 系统能量函数
            sampling_strategy: 采样策略
            num_samples: 采样数量
            tolerance: 容差
            
        Returns:
            系统验证结果
        """
        if system_energy_func is None:
            system_energy_func = self.energy_function
        if system_energy_func is None:
            raise ValueError("请提供系统能量函数")
        
        print(f"开始系统级Hessian验证...")
        print(f"采样策略: {sampling_strategy}, 样本数: {num_samples}")
        
        # 生成采样点
        sample_points = self._generate_sample_points(sampling_strategy, num_samples)
        
        # 使用点验证器进行批量验证
        batch_results = self.point_validator.batch_validate(sample_points, system_energy_func)
        
        # 系统级统计分析
        system_stats = self._compute_system_statistics(batch_results)
        
        # 检查系统一致性
        system_consistent = self._check_system_consistency(system_stats, tolerance)
        
        results = {
            'system_consistent': system_consistent,
            'sampling_strategy': sampling_strategy,
            'num_samples': num_samples,
            'batch_results': batch_results,
            'system_statistics': system_stats,
            'consistency_check_passed': system_consistent
        }
        
        self.print_system_validation_summary(results)
        return results
    
    def _generate_sample_points(self, strategy: str, num_points: int) -> List[np.ndarray]:
        """
        根据策略生成采样点
        """
        if self.vertices is None:
            raise ValueError("请先加载顶点数据")
        
        n_vertices = len(self.vertices)
        dim = self.vertices.shape[1] if self.vertices.ndim > 1 else 1
        
        if strategy == 'random':
            # 随机采样
            points = []
            for _ in range(num_points):
                if dim == 1:
                    point = np.random.rand(n_vertices)
                else:
                    point = np.random.rand(n_vertices, dim)
                points.append(point)
            return points
            
        elif strategy == 'uniform_grid':
            # 均匀网格采样（简化版）
            points = []
            grid_values = np.linspace(0, 1, int(num_points ** (1/dim)) + 1)
            # 这里需要根据具体维度实现网格生成
            return points[:num_points]
            
        elif strategy == 'critical_points':
            # 关键点采样（如碰撞点附近）
            points = []
            if self.collision_pairs:
                for pair in self.collision_pairs[:num_points]:
                    # 在碰撞对中点附近采样
                    mid_point = (self.vertices[pair[0]] + self.vertices[pair[1]]) / 2
                    points.append(mid_point)
            return points
            
        else:
            raise ValueError(f"不支持的采样策略: {strategy}")
    
    def _compute_system_statistics(self, batch_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        计算系统级统计量
        """
        detailed_results = batch_results['detailed_results']
        
        errors_analytical_fd = [r['error_analytical_vs_finite_diff'] for r in detailed_results]
        errors_analytical_ad = [r['error_analytical_vs_auto_diff'] for r in detailed_results]
        errors_fd_ad = [r['error_finite_diff_vs_auto_diff'] for r in detailed_results]
        
        stats = {
            'mean_error_analytical_fd': np.mean(errors_analytical_fd),
            'max_error_analytical_fd': np.max(errors_analytical_fd),
            'mean_error_analytical_ad': np.mean(errors_analytical_ad),
            'max_error_analytical_ad': np.max(errors_analytical_ad),
            'mean_error_fd_ad': np.mean(errors_fd_ad),
            'max_error_fd_ad': np.max(errors_fd_ad),
            'error_std_analytical_fd': np.std(errors_analytical_fd),
            'pass_rate': batch_results['pass_rate']
        }
        
        return stats
    
    def _check_system_consistency(self, stats: Dict[str, Any], tolerance: float) -> bool:
        """
        检查系统一致性
        """
        consistency_checks = [
            stats['mean_error_analytical_fd'] < tolerance,
            stats['mean_error_analytical_ad'] < tolerance,
            stats['mean_error_fd_ad'] < tolerance,
            stats['pass_rate'] > 0.8  # 80%以上的点要通过
        ]
        
        return all(consistency_checks)
    
    def validate_hessian_symmetry(self, test_points: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        验证Hessian矩阵的对称性
        """
        if test_points is None:
            test_points = self._generate_sample_points('random', 5)
        
        symmetry_errors = []
        for point in test_points:
            hessian = self.point_validator._compute_analytical_hessian(point, self.energy_function)
            symmetry_error = self.compute_norm(hessian - hessian.T) / self.compute_norm(hessian)
            symmetry_errors.append(symmetry_error)
        
        results = {
            'mean_symmetry_error': np.mean(symmetry_errors),
            'max_symmetry_error': np.max(symmetry_errors),
            'symmetry_passed': np.max(symmetry_errors) < 1e-10
        }
        
        return results
    
    def print_system_validation_summary(self, results: Dict[str, Any]):
        """打印系统验证摘要"""
        stats = results['system_statistics']
        
        print("\n" + "="*60)
        print("系统级Hessian验证结果摘要")
        print("="*60)
        print(f"系统一致性: {'通过' if results['system_consistent'] else '失败'}")
        print(f"采样点数: {results['num_samples']}")
        print(f"通过率: {stats['pass_rate']:.1%}")
        print(f"平均误差 (解析vs有限差分): {stats['mean_error_analytical_fd']:.6e}")
        print(f"平均误差 (解析vs自动微分): {stats['mean_error_analytical_ad']:.6e}")
        print(f"最大误差: {stats['max_error_analytical_fd']:.6e}")
        print("="*60)