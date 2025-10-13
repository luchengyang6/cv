import numpy as np
from .knn import KNeighborsClassifier
from .utils import cross_validate


def grid_search_knn(X, y, param_grid, cv=5):
    """
    K-NN网格搜索参数优化

    参数:
        X: 特征数据
        y: 标签数据
        param_grid: 参数网格
        cv: 交叉验证折数

    返回:
        dict: 最佳参数和结果
    """
    best_score = 0
    best_params = {}
    results = []

    # 遍历所有参数组合
    for k in param_grid['n_neighbors']:
        for metric in param_grid['metric']:
            for weight in param_grid['weights']:
                model = KNeighborsClassifier(
                    n_neighbors=k,
                    metric=metric,
                    weights=weight
                )

                # 交叉验证
                scores = cross_validate(model, X, y, cv=cv)
                mean_score = np.mean(scores)

                result = {
                    'params': {'n_neighbors': k, 'metric': metric, 'weights': weight},
                    'mean_score': mean_score,
                    'std_score': np.std(scores),
                    'cv_scores': scores
                }
                results.append(result)

                print(f"参数: k={k}, metric={metric}, weights={weight}, "
                      f"平均准确率: {mean_score:.4f} ± {np.std(scores):.4f}")

                if mean_score > best_score:
                    best_score = mean_score
                    best_params = result['params']

    return {
        'best_score': best_score,
        'best_params': best_params,
        'all_results': results
    }