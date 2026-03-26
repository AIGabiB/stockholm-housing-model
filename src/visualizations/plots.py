import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importance(feature_importance, title = "Feature Importance"):

    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor("#0E1117")     
    ax.set_facecolor("#0E1117") 

    ax.barh(feature_importance.index, feature_importance.values, color="#4CAF50")

    ax.set_title(title, fontweight="bold", fontsize=16, color="#E0E0E0")
    ax.set_xlabel("Importance", fontweight="bold", color="#E0E0E0")
    ax.set_ylabel("Features", fontweight="bold", color="#E0E0E0")

    ax.tick_params(axis='x', colors="#E0E0E0")
    ax.tick_params(axis='y', colors="#E0E0E0")

    for spine in ax.spines.values():
        spine.set_edgecolor("white")
    
    return fig

def plot_scatter(ytest,ypred ,title = "Actual price vs Predicted price"):
    np.random.seed(42)
    fig = plt.figure(facecolor="#0E1117",figsize=[5,3])

    ax = plt.gca()
    ax.set_facecolor("#0E1117")
    ax.tick_params(colors='#E0E0E0', which='both')
    for spine in ax.spines.values():
        spine.set_color('#E0E0E0')
    
    plt.scatter(ytest,ypred,zorder=2,color="#4CAF50")
    plt.plot([1,30000000],[1,30000000],color="red",ls="dashed",alpha=0.8)
    plt.xlabel("Actual price",c = "#E0E0E0")
    plt.ylabel("Predicted price",c = "#E0E0E0")
    plt.title(title,c = "#E0E0E0",fontweight="bold",fontsize=16)
    plt.grid(True,alpha=0.1)

    return fig 