import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

class LightweightHypergraphTemporal(nn.Module):
    """Compact event classifier with hypergraph evidence and low-rank temporal mixing."""
    def __init__(self, input_dim, hidden=64, low_rank=16, n_classes=2):
        super().__init__()
        self.event_proj = nn.Linear(input_dim, hidden)
        self.hyper_proj = nn.Linear(input_dim, hidden)
        self.temporal_down = nn.Linear(hidden, low_rank, bias=False)
        self.temporal_up = nn.Linear(low_rank, hidden, bias=False)
        self.classifier = nn.Sequential(nn.ReLU(), nn.Dropout(0.15), nn.Linear(hidden, n_classes))
        self.aux_head = nn.Linear(hidden, 1)

    def forward(self, x, hyper):
        h = torch.relu(self.event_proj(x) + self.hyper_proj(hyper))
        # Low-rank feature mixing is a lightweight temporal-context proxy.
        # For true temporal windows, supply lagged features as part of x.
        z = self.temporal_up(torch.tanh(self.temporal_down(h)))
        fused = h + z
        return self.classifier(fused), torch.sigmoid(self.aux_head(fused)).squeeze(-1)

def train_model(Xtr, Htr, ytr, Xva, Hva, yva, epochs=20, batch_size=256, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    classes = np.unique(ytr)
    class_map = {v:i for i,v in enumerate(classes)}
    yt = np.array([class_map[v] for v in ytr], dtype=np.int64)
    yv = np.array([class_map.get(v,0) for v in yva], dtype=np.int64)
    model = LightweightHypergraphTemporal(Xtr.shape[1], n_classes=len(classes)).to(device)
    ds = TensorDataset(torch.tensor(Xtr), torch.tensor(Htr), torch.tensor(yt))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train(); losses=[]
        for x,h,y in loader:
            x,h,y=x.to(device),h.to(device),y.to(device)
            logits, aux = model(x,h)
            loss = loss_fn(logits,y) + 0.05 * ((aux - (y>0).float())**2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); losses.append(loss.item())
        print(f"Epoch {ep+1:02d}/{epochs} loss={np.mean(losses):.4f}")
    model.eval()
    with torch.no_grad():
        logits,_=model(torch.tensor(Xva,device=device),torch.tensor(Hva,device=device))
        probs=torch.softmax(logits,dim=1).cpu().numpy()
    return model, probs, classes
