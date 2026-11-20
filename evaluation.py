from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,
                             matthews_corrcoef,confusion_matrix,roc_auc_score,
                             average_precision_score)
def evaluate_classifier(y_true,y_pred,probabilities,classes,out_dir="results"):
    out=Path(out_dir); (out/"graphs").mkdir(parents=True,exist_ok=True)
    positive=1 if len(classes)>1 else 0
    yt=np.array([1 if v==classes[positive] else 0 for v in y_true])
    yp=np.array([1 if v==classes[positive] else 0 for v in y_pred])
    p=probabilities[:,positive] if probabilities.shape[1]>1 else probabilities[:,0]
    metrics={"accuracy":accuracy_score(yt,yp),"precision":precision_score(yt,yp,zero_division=0),
             "recall":recall_score(yt,yp,zero_division=0),"f1":f1_score(yt,yp,zero_division=0),
             "mcc":matthews_corrcoef(yt,yp),
             "fpr":confusion_matrix(yt,yp,labels=[0,1])[0,1]/max(1,(yt==0).sum())}
    if len(np.unique(yt))>1:
        metrics["auroc"]=roc_auc_score(yt,p); metrics["auprc"]=average_precision_score(yt,p)
    import json
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2))
    cm=confusion_matrix(yt,yp,labels=[0,1])
    plt.figure(); plt.imshow(cm); plt.title("Threat Detection Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.colorbar()
    plt.tight_layout(); plt.savefig(out/"graphs/confusion_matrix.png",dpi=200); plt.close()
    return metrics
