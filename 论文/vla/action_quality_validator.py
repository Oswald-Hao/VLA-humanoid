#!/usr/bin/env python3
"""
动作质量验证系统
用于实时监控和评估模型输出的动作质量
"""

import numpy as np
import torch
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import matplotlib.pyplot as plt

class ActionQualityValidator:
    """动作质量验证器"""
    
    def __init__(self):
        # 关节限制（基于实际机器人限制）
        self.joint_limits = np.array([
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,  # 关节1-7
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,  # 关节8-14
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,  # 关节15-21
            2.8973, 1.7628, 2.8973, -0.0698, 2.8973                   # 关节22-26
        ])
        
        # 合理的物理约束
        self.reasonable_velocity = 0.15      # 合理的关节速度 (rad/step)
        self.reasonable_acceleration = 0.08  # 合理的关节加速度 (rad/step²)
        self.reasonable_jerk = 0.05          # 合理的加加速度 (rad/step³)
        
        # 关节组定义（用于协调性检查）
        self.joint_groups = {
            'left_arm': [12, 13, 14],      # 左臂主要关节
            'right_arm': [19, 20, 21],    # 右臂主要关节
            'left_hand': [15, 16, 17],    # 左手关节
            'right_hand': [22, 23, 24],   # 右手关节
            'torso': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # 躯干关节
        }
        
        # 质量标准（基于真实数据分析）
        self.quality_standards = {
            'wave': {
                'expected_variance': 0.108,
                'key_joints': [12, 14, 13, 15, 18],
                'expected_coordination': 0.7  # 左臂协调性期望值
            },
            'welcome': {
                'expected_variance': 0.111,
                'key_joints': [12, 22, 15, 14, 19],
                'expected_coordination': 0.6  # 双臂协调性期望值
            }
        }
    
    def validate_action(self, predicted_action: np.ndarray, 
                       target_action: Optional[np.ndarray] = None,
                       instruction: Optional[str] = None) -> Dict[str, float]:
        """
        全面验证动作质量
        
        Args:
            predicted_action: 预测的动作序列 [seq_len, action_dim]
            target_action: 目标动作序列 [seq_len, action_dim] (可选)
            instruction: 指令类型 (可选)
            
        Returns:
            质量验证结果字典
        """
        validation_results = {}
        
        # 1. 基础物理约束检查
        validation_results['within_limits'] = self._check_joint_limits(predicted_action)
        validation_results['smoothness'] = self._check_smoothness(predicted_action)
        validation_results['reasonable_velocity'] = self._check_velocity(predicted_action)
        validation_results['reasonable_acceleration'] = self._check_acceleration(predicted_action)
        validation_results['reasonable_jerk'] = self._check_jerk(predicted_action)
        
        # 2. 动作特征分析
        validation_results['action_variance'] = self._analyze_action_variance(predicted_action)
        validation_results['action_consistency'] = self._check_action_consistency(predicted_action)
        
        # 3. 关节协调性检查
        validation_results['joint_coordination'] = self._check_joint_coordination(predicted_action)
        
        # 4. 如果有目标动作，计算相似度
        if target_action is not None:
            validation_results['pattern_similarity'] = self._check_pattern_similarity(
                predicted_action, target_action
            )
            validation_results['magnitude_match'] = self._check_magnitude_match(
                predicted_action, target_action
            )
        
        # 5. 如果有指令信息，进行指令特定的验证
        if instruction is not None:
            instruction_results = self._validate_instruction_specific(
                predicted_action, instruction
            )
            validation_results.update(instruction_results)
        
        # 6. 计算综合质量分数
        validation_results['overall_quality'] = self._compute_overall_quality(validation_results)
        
        return validation_results
    
    def _check_joint_limits(self, action: np.ndarray) -> float:
        """检查关节限制"""
        max_action = np.max(np.abs(action), axis=0)
        violations = np.sum(max_action > self.joint_limits)
        return 1.0 - (violations / len(self.joint_limits))  # 返回合规比例
    
    def _check_smoothness(self, action: np.ndarray) -> float:
        """检查动作平滑性"""
        if action.shape[0] < 3:
            return 1.0
        
        # 计算二阶差分（加速度）
        acceleration = np.diff(np.diff(action, axis=0), axis=0)
        acceleration_magnitude = np.mean(np.abs(acceleration))
        
        # 平滑性分数：加速度越小越平滑
        smoothness = np.exp(-acceleration_magnitude / self.reasonable_acceleration)
        return min(1.0, smoothness)
    
    def _check_velocity(self, action: np.ndarray) -> float:
        """检查速度合理性"""
        if action.shape[0] < 2:
            return 1.0
        
        velocity = np.diff(action, axis=0)
        velocity_magnitude = np.mean(np.abs(velocity))
        
        # 速度合理性分数
        velocity_score = np.exp(-velocity_magnitude / self.reasonable_velocity)
        return min(1.0, velocity_score)
    
    def _check_acceleration(self, action: np.ndarray) -> float:
        """检查加速度合理性"""
        if action.shape[0] < 3:
            return 1.0
        
        acceleration = np.diff(np.diff(action, axis=0), axis=0)
        acceleration_magnitude = np.mean(np.abs(acceleration))
        
        acceleration_score = np.exp(-acceleration_magnitude / self.reasonable_acceleration)
        return min(1.0, acceleration_score)
    
    def _check_jerk(self, action: np.ndarray) -> float:
        """检查加加速度合理性"""
        if action.shape[0] < 4:
            return 1.0
        
        jerk = np.diff(np.diff(np.diff(action, axis=0), axis=0), axis=0)
        jerk_magnitude = np.mean(np.abs(jerk))
        
        jerk_score = np.exp(-jerk_magnitude / self.reasonable_jerk)
        return min(1.0, jerk_score)
    
    def _analyze_action_variance(self, action: np.ndarray) -> float:
        """分析动作变化量"""
        joint_variance = np.var(action, axis=0)
        overall_variance = np.mean(joint_variance)
        
        # 变化量合理性分数（基于真实数据经验）
        if overall_variance < 0.001:
            return 0.1  # 变化太小
        elif overall_variance > 1.0:
            return 0.5  # 变化太大
        else:
            return 1.0  # 合理变化
    
    def _check_action_consistency(self, action: np.ndarray) -> float:
        """检查动作一致性"""
        # 检查动作是否有异常跳跃
        if action.shape[0] < 2:
            return 1.0
        
        diff = np.diff(action, axis=0)
        # 检查是否有异常大的跳跃
        max_jump = np.max(np.abs(diff))
        consistency_score = np.exp(-max_jump / 0.5)  # 0.5是可接受的跳跃阈值
        return min(1.0, consistency_score)
    
    def _check_joint_coordination(self, action: np.ndarray) -> float:
        """检查关节协调性"""
        coordination_scores = []
        
        for group_name, joint_indices in self.joint_groups.items():
            if len(joint_indices) < 2:
                continue
            
            group_data = action[:, joint_indices]
            
            # 计算组内关节的相关性
            correlations = []
            for i in range(len(joint_indices)):
                for j in range(i+1, len(joint_indices)):
                    if np.std(group_data[:, i]) > 0 and np.std(group_data[:, j]) > 0:
                        corr = np.corrcoef(group_data[:, i], group_data[:, j])[0, 1]
                        correlations.append(abs(corr))
            
            if correlations:
                group_coordination = np.mean(correlations)
                coordination_scores.append(group_coordination)
        
        return np.mean(coordination_scores) if coordination_scores else 0.5
    
    def _check_pattern_similarity(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """检查动作模式相似性"""
        if predicted.shape != target.shape:
            return 0.0
        
        # 计算每个关节的相关性
        correlations = []
        for joint_idx in range(predicted.shape[1]):
            if np.std(target[:, joint_idx]) > 0:
                corr = np.corrcoef(predicted[:, joint_idx], target[:, joint_idx])[0, 1]
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _check_magnitude_match(self, predicted: np.ndarray, target: np.ndarray) -> float:
        """检查幅度匹配度"""
        pred_std = np.std(predicted, axis=0)
        target_std = np.std(target, axis=0)
        
        # 计算相对幅度差异
        relative_diff = np.abs(pred_std - target_std) / (target_std + 1e-8)
        magnitude_match = np.exp(-np.mean(relative_diff))
        
        return min(1.0, magnitude_match)
    
    def _validate_instruction_specific(self, action: np.ndarray, instruction: str) -> Dict[str, float]:
        """指令特定的验证"""
        results = {}
        
        if instruction not in self.quality_standards:
            return results
        
        standard = self.quality_standards[instruction]
        
        # 检查变化量是否符合期望
        joint_variance = np.var(action, axis=0)
        overall_variance = np.mean(joint_variance)
        expected_variance = standard['expected_variance']
        
        variance_match = np.exp(-abs(overall_variance - expected_variance) / expected_variance)
        results['variance_match'] = min(1.0, variance_match)
        
        # 检查关键关节是否正确
        key_joints = standard['key_joints']
        key_joint_variance = np.mean([joint_variance[j] for j in key_joints])
        other_joint_variance = np.mean([joint_variance[j] for j in range(len(joint_variance)) if j not in key_joints])
        
        # 关键关节的变化量应该大于其他关节
        if other_joint_variance > 0:
            key_joint_importance = key_joint_variance / other_joint_variance
        else:
            key_joint_importance = 2.0 if key_joint_variance > 0 else 1.0
        
        results['key_joint_importance'] = min(1.0, key_joint_importance / 2.0)  # 归一化到0-1
        
        return results
    
    def _compute_overall_quality(self, validation_results: Dict[str, float]) -> float:
        """计算综合质量分数"""
        # 定义各项指标的权重
        weights = {
            'within_limits': 0.15,
            'smoothness': 0.1,
            'reasonable_velocity': 0.1,
            'reasonable_acceleration': 0.1,
            'reasonable_jerk': 0.05,
            'action_variance': 0.1,
            'action_consistency': 0.1,
            'joint_coordination': 0.1,
            'pattern_similarity': 0.1,
            'magnitude_match': 0.1
        }
        
        overall_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in validation_results:
                overall_score += validation_results[metric] * weight
                total_weight += weight
        
        return overall_score / total_weight if total_weight > 0 else 0.0


class TrainingQualityMonitor:
    """训练质量监控器"""
    
    def __init__(self, validator: ActionQualityValidator):
        self.validator = validator
        self.quality_history = []
        self.epoch_quality = []
        self.instruction_quality = defaultdict(list)
    
    def monitor_batch(self, predicted_actions: torch.Tensor, 
                     target_actions: torch.Tensor, 
                     instruction_ids: torch.Tensor) -> Dict[str, float]:
        """监控每个batch的质量"""
        batch_quality = []
        
        # 转换为numpy数组
        pred_actions = predicted_actions.detach().cpu().numpy()
        target_actions = target_actions.detach().cpu().numpy()
        instruction_ids_np = instruction_ids.detach().cpu().numpy()
        
        id_to_instruction = {0: 'wave', 1: 'welcome'}
        
        for i in range(len(pred_actions)):
            pred_action = pred_actions[i]
            target_action = target_actions[i]
            instruction_id = instruction_ids_np[i]
            instruction = id_to_instruction[instruction_id]
            
            quality_metrics = self.validator.validate_action(
                pred_action, target_action, instruction
            )
            batch_quality.append(quality_metrics)
            
            # 记录指令特定的质量
            self.instruction_quality[instruction].append(quality_metrics)
        
        # 计算batch平均质量
        avg_quality = self._compute_average_quality(batch_quality)
        self.quality_history.append(avg_quality)
        
        return avg_quality
    
    def monitor_epoch(self, epoch: int) -> Dict[str, float]:
        """监控每个epoch的质量"""
        if not self.quality_history:
            return {}
        
        # 计算epoch平均质量
        epoch_avg_quality = self._compute_average_quality(self.quality_history)
        self.epoch_quality.append((epoch, epoch_avg_quality))
        
        return epoch_avg_quality
    
    def _compute_average_quality(self, quality_list: List[Dict[str, float]]) -> Dict[str, float]:
        """计算平均质量"""
        if not quality_list:
            return {}
        
        avg_quality = {}
        for key in quality_list[0].keys():
            values = [q[key] for q in quality_list if key in q]
            if values:
                avg_quality[key] = np.mean(values)
        
        return avg_quality
    
    def get_quality_report(self) -> str:
        """生成质量报告"""
        if not self.quality_history:
            return "No quality data available"
        
        report = "Training Quality Report\n"
        report += "=" * 50 + "\n"
        
        # 计算各项指标的平均值
        latest_quality = self.quality_history[-1]
        for key, value in latest_quality.items():
            report += f"{key}: {value:.4f}\n"
        
        # 添加指令特定的质量信息
        report += "\nInstruction-Specific Quality:\n"
        for instruction, qualities in self.instruction_quality.items():
            if qualities:
                latest = qualities[-1]
                report += f"{instruction}: {latest.get('overall_quality', 0):.4f}\n"
        
        # 质量趋势
        if len(self.quality_history) > 10:
            recent_qualities = [q.get('overall_quality', 0) for q in self.quality_history[-10:]]
            trend = "improving" if recent_qualities[-1] > recent_qualities[0] else "declining"
            report += f"\nQuality Trend (last 10 batches): {trend}\n"
        
        return report
    
    def save_quality_history(self, filepath: str):
        """保存质量历史"""
        # 转换numpy数据类型为Python原生类型
        def convert_to_serializable(obj):
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64, np.float16)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8)):
                return int(obj)
            elif isinstance(obj, (np.uint32, np.uint64, np.uint16, np.uint8)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, torch.Tensor):
                return obj.detach().cpu().numpy().tolist()
            else:
                return obj
        
        quality_data = {
            'quality_history': convert_to_serializable(self.quality_history),
            'epoch_quality': convert_to_serializable(self.epoch_quality),
            'instruction_quality': convert_to_serializable(dict(self.instruction_quality))
        }
        
        with open(filepath, 'w') as f:
            json.dump(quality_data, f, indent=2)


def test_quality_validator():
    """测试质量验证器"""
    print("测试动作质量验证器...")
    
    validator = ActionQualityValidator()
    
    # 创建测试数据
    seq_len, action_dim = 32, 26
    
    # 测试1：高质量动作
    good_action = np.random.randn(seq_len, action_dim) * 0.1
    good_action = np.cumsum(good_action, axis=0)  # 积分得到平滑轨迹
    
    # 测试2：低质量动作
    bad_action = np.random.randn(seq_len, action_dim) * 2.0
    
    # 验证动作质量
    good_quality = validator.validate_action(good_action, instruction='wave')
    bad_quality = validator.validate_action(bad_action, instruction='wave')
    
    print(f"高质量动作分数: {good_quality['overall_quality']:.4f}")
    print(f"低质量动作分数: {bad_quality['overall_quality']:.4f}")
    
    # 检查各项指标
    print("\n高质量动作指标:")
    for key, value in good_quality.items():
        print(f"  {key}: {value:.4f}")
    
    print("\n验证器测试完成！")


if __name__ == "__main__":
    test_quality_validator()