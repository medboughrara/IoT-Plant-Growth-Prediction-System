import matplotlib.pyplot as plt
import seaborn as sns

def plot_correlation_matrix(correlation_matrix, title='Feature Correlation Matrix'):
    """Plot correlation matrix heatmap"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(title)
    plt.tight_layout()
    return plt.gcf()

def plot_feature_importance(feature_importance, title='Feature Importance for Plant Classification'):
    """Plot feature importance bar chart"""
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='importance', y='feature')
    plt.title(title)
    plt.xlabel('Importance Score')
    plt.tight_layout()
    return plt.gcf()

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix'):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    return plt.gcf()
