#!/usr/bin/env python3
"""
评估脚本 - 从已训练模型评估性能

功能：
- 加载训练好的模型checkpoint
- 在测试集上推理
- 计算准确率、AUC、灵敏度、特异性等指标
- 支持单图谱和双图谱模型

无需重新训练，直接使用.pth文件
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tna.data.dataset import TNADataset, DualAtlasTNADataset, DualAtlasTNASubset
from tna.data.splits import train_test_splitKFold
from tna.models import TNA, DualAtlasTNA
from tna.training.metrics import compute_classification_metrics
from tna.configs.model_config import TNAConfig
from tna.configs.path_config import PathConfig
from tna.configs.atlas_config import get_atlas_config
from torch_geometric.loader import DataLoader


def load_model_from_checkpoint(checkpoint_path, config, device='cuda'):
    """
    从checkpoint加载模型
    
    参数:
        checkpoint_path: 模型文件路径 (.pth)
        config: 模型配置
        device: 计算设备
    
    返回:
        model: 加载好的模型
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 提取模型状态字典
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # 创建模型
    model_kwargs = {
        'd_model': config.dim_hidden,
        'num_heads': config.num_heads,
        'dim_feedforward': config.dim_hidden * 4,
        'dropout': config.dropout,
        'num_layers': config.num_layers,
        'batch_norm': True,
        'pe': config.pe is not None and config.pe != "",
        'pe_dim': config.pe_dim,
        'gnn_type': config.gnn_type,
        'se': config.se,
        'use_edge_attr': config.use_edge_attr,
        'num_edge_features': 2,
        'edge_dim': config.edge_dim,
        'use_gnn': config.use_gnn if hasattr(config, 'use_gnn') else True,
        'use_attention': config.use_attention if hasattr(config, 'use_attention') else True,
        'use_hierarchical_graph': config.use_hierarchical_graph if hasattr(config, 'use_hierarchical_graph') else True,
    }
    
    if config.dual_atlas:
        model = DualAtlasTNA(num_class=2, **model_kwargs)
    else:
        atlas_cfg = get_atlas_config(config.atlas)
        model = TNA(
            in_size=atlas_cfg['num_nodes'],
            num_class=2,
            num_nodes=atlas_cfg['num_nodes'],
            comm_boundaries=atlas_cfg['comm_boundaries'],
            **model_kwargs
        )
    
    # 加载权重
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, test_loader, device='cuda', is_dual_atlas=False):
    """
    评估模型
    
    参数:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        is_dual_atlas: 是否为双图谱模型
    
    返回:
        metrics: 评估指标字典
    """
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for data in test_loader:
            # 前向传播
            if is_dual_atlas:
                data_cc200, data_aal116 = data
                data_cc200 = data_cc200.to(device)
                data_aal116 = data_aal116.to(device)
                
                output = model(data_cc200, data_aal116)
                labels = data_cc200.y
            else:
                data = data.to(device)
                output = model(data)
                labels = data.y
            
            # 预测
            probabilities = F.softmax(output, dim=1)[:, 1]  # 正类概率
            predictions = torch.argmax(output, dim=1)
            
            # 收集结果
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())
    
    # 计算指标
    metrics = compute_classification_metrics(all_labels, all_predictions, all_probabilities)
    
    # 添加准确率
    accuracy = sum(p == l for p, l in zip(all_predictions, all_labels)) / len(all_labels)
    metrics['accuracy'] = accuracy
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='评估已训练的TNA模型')
    
    # 模型checkpoint目录
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                        help='包含.pth文件的目录路径')
    parser.add_argument('--fold', type=int, default=None,
                        help='指定评估某一折 (1-10)，不指定则评估所有折')
    
    # 数据配置
    parser.add_argument('--base_dir', type=str, required=True,
                        help='项目根目录（数据与日志所在路径）')
    parser.add_argument('--dataset', type=str, default=None,
                        help='数据集名称')
    parser.add_argument('--atlas', type=str, default=None,
                        help='图谱名称 (cc200 或 aal116)')
    parser.add_argument('--dual_atlas', action='store_true',
                        help='使用双图谱模型')
    parser.add_argument('--kfold', type=int, default=None,
                        help='K折数量')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU设备ID')
    
    args = parser.parse_args()

    defaults = TNAConfig()
    dataset = args.dataset if args.dataset is not None else defaults.dataset
    atlas = args.atlas if args.atlas is not None else defaults.atlas
    kfold = args.kfold if args.kfold is not None else defaults.Kfold
    batch_size = args.batch_size if args.batch_size is not None else defaults.batch_size
    gpu = args.gpu if args.gpu is not None else 0

    device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    config = TNAConfig()
    config.update(
        dataset=dataset,
        atlas=atlas,
        dual_atlas=args.dual_atlas,
        Kfold=kfold,
        batch_size=batch_size
    )
    
    path_config = PathConfig(base_dir=args.base_dir)
    
    # 打印配置
    print("\n" + "=" * 80)
    print("评估配置")
    print("=" * 80)
    print(f"数据集: {dataset}")
    print(f"图谱: {atlas}")
    print(f"双图谱: {args.dual_atlas}")
    print(f"K折: {kfold}")
    print(f"Checkpoint目录: {args.checkpoint_dir}")
    print(f"Base目录: {args.base_dir}")
    print("=" * 80)
    
    # 加载数据集
    print("\n加载数据集...")
    
    if args.dual_atlas:
        full_dataset = DualAtlasTNADataset(
            root=path_config.data_dir,
            dataset_name=config.dataset.lower().replace('-', '_'),
            atlas_cc200='cc200',
            atlas_aal116='aal116'
        )
        print(f"✓ 双图谱数据集加载完成: {len(full_dataset)} 个样本")
    else:
        full_dataset = TNADataset(
            root=path_config.data_dir,
            dataset_name=config.dataset.lower().replace('-', '_'),
            atlas_name=config.atlas
        )
        print(f"✓ {config.atlas.upper()} 数据集加载完成: {len(full_dataset)} 个样本")
    
    # K折划分
    num_samples = len(full_dataset)
    kfold_splits = train_test_splitKFold(
        kfold=config.Kfold,
        random_state=42,
        n_sub=num_samples
    )
    
    # 确定要评估的折数
    if args.fold is not None:
        if args.fold < 1 or args.fold > args.kfold:
            raise ValueError(f"无效的fold编号: {args.fold}. 必须在1-{args.kfold}之间")
        folds_to_evaluate = [args.fold - 1]
        print(f"\n将评估 Fold {args.fold}")
    else:
        folds_to_evaluate = range(args.kfold)
        print(f"\n将评估所有 {args.kfold} 折")
    
    # 用于统计的列表
    acc_list = []
    auc_list = []
    sensitivity_list = []
    specificity_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    
    # 评估每一折
    for fold_idx in folds_to_evaluate:
        print("\n" + "=" * 80)
        print(f"评估 Fold {fold_idx + 1}/{config.Kfold}")
        print("=" * 80)
        
        # 检查checkpoint是否存在
        checkpoint_path = os.path.join(args.checkpoint_dir, f"model_fold{fold_idx + 1}.pth")
        if not os.path.exists(checkpoint_path):
            print(f"✗ Checkpoint不存在: {checkpoint_path}")
            continue
        
        # 加载模型
        print(f"加载模型: {checkpoint_path}")
        model = load_model_from_checkpoint(checkpoint_path, config, device)
        print(f"✓ 模型加载成功")
        
        # 创建测试集
        train_idx, test_idx = kfold_splits[fold_idx]
        
        if args.dual_atlas:
            train_dataset = DualAtlasTNASubset(full_dataset, train_idx)
            test_dataset = DualAtlasTNASubset(full_dataset, test_idx)
        else:
            train_dataset = full_dataset[train_idx]
            test_dataset = full_dataset[test_idx]
        
        test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
        
        print(f"测试集样本数: {len(test_dataset)}")
        
        # 评估
        print("开始评估...")
        metrics = evaluate_model(model, test_loader, device, is_dual_atlas=args.dual_atlas)
        
        # 打印结果
        print(f"\nFold {fold_idx + 1} 评估结果:")
        print(f"  准确率 (Accuracy):    {metrics['accuracy']:.4f}")
        print(f"  AUC:                  {metrics.get('auc', 0):.4f}")
        print(f"  灵敏度 (Sensitivity): {metrics.get('sensitivity', 0):.4f}")
        print(f"  特异性 (Specificity): {metrics.get('specificity', 0):.4f}")
        print(f"  精确率 (Precision):   {metrics.get('precision', 0):.4f}")
        print(f"  召回率 (Recall):      {metrics.get('recall', 0):.4f}")
        print(f"  F1分数:               {metrics.get('f1', 0):.4f}")
        
        # 收集指标
        acc_list.append(metrics['accuracy'])
        auc_list.append(metrics.get('auc', 0))
        sensitivity_list.append(metrics.get('sensitivity', 0))
        specificity_list.append(metrics.get('specificity', 0))
        precision_list.append(metrics.get('precision', 0))
        recall_list.append(metrics.get('recall', 0))
        f1_list.append(metrics.get('f1', 0))
    
    # 如果评估了多个fold，计算平均值
    if len(acc_list) > 1:
        print("\n" + "=" * 80)
        print("K折交叉验证汇总结果")
        print("=" * 80)
        
        print(f"\n各折准确率: {[f'{a:.4f}' for a in acc_list]}")
        
        # 计算统计量
        avg_acc = np.mean(acc_list)
        std_acc = np.std(acc_list)
        avg_auc = np.mean(auc_list)
        std_auc = np.std(auc_list)
        avg_sen = np.mean(sensitivity_list)
        std_sen = np.std(sensitivity_list)
        avg_spec = np.mean(specificity_list)
        std_spec = np.std(specificity_list)
        avg_prec = np.mean(precision_list)
        std_prec = np.std(precision_list)
        avg_recall = np.mean(recall_list)
        std_recall = np.std(recall_list)
        avg_f1 = np.mean(f1_list)
        std_f1 = np.std(f1_list)
        
        # 打印汇总结果
        print(f"\n汇总结果:")
        print(f"  准确率:    {avg_acc:.4f} ± {std_acc:.4f}")
        print(f"  AUC:       {avg_auc:.4f} ± {std_auc:.4f}")
        print(f"  灵敏度:    {avg_sen:.4f} ± {std_sen:.4f}")
        print(f"  特异性:    {avg_spec:.4f} ± {std_spec:.4f}")
        print(f"  精确率:    {avg_prec:.4f} ± {std_prec:.4f}")
        print(f"  召回率:    {avg_recall:.4f} ± {std_recall:.4f}")
        print(f"  F1分数:    {avg_f1:.4f} ± {std_f1:.4f}")
        
        # 保存评估结果
        results = {
            'avg_accuracy': avg_acc,
            'std_accuracy': std_acc,
            'avg_auc': avg_auc,
            'std_auc': std_auc,
            'avg_sensitivity': avg_sen,
            'std_sensitivity': std_sen,
            'avg_specificity': avg_spec,
            'std_specificity': std_spec,
            'avg_precision': avg_prec,
            'std_precision': std_prec,
            'avg_recall': avg_recall,
            'std_recall': std_recall,
            'avg_f1': avg_f1,
            'std_f1': std_f1,
            'fold_results': {
                'accuracy': acc_list,
                'auc': auc_list,
                'sensitivity': sensitivity_list,
                'specificity': specificity_list,
                'precision': precision_list,
                'recall': recall_list,
                'f1': f1_list,
            }
        }
        
        # 保存为JSON
        import json
        eval_results_path = os.path.join(args.checkpoint_dir, 'evaluation_results.json')
        with open(eval_results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ 评估结果已保存: {eval_results_path}")
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)


if __name__ == '__main__':
    main()

