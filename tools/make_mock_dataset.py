import os, json, numpy as np
np.random.seed(0)
root="dataset"; os.makedirs(root, exist_ok=True)
for i in range(1,41):
    d=os.path.join(root,f"seq_{i:04d}"); os.makedirs(d, exist_ok=True)
    T=64; tac=np.random.rand(T,16,16).astype("float32")
    np.save(os.path.join(d,"tactile.npy"), tac)
    if i%2==0:
        phrases=["软","轻压","不打滑"]; sentences=["轻压在软表面，基本不打滑"]
    else:
        phrases=["硬","中等按压","轻微打滑"]; sentences=["中等按压在硬表面，有轻微打滑"]
    json.dump({"phrases":phrases,"sentences":sentences}, open(os.path.join(d,"text.json"),"w"), ensure_ascii=False, indent=2)
    json.dump({"timestamps":{"tactile":list(range(T))}}, open(os.path.join(d,"meta.json"),"w"))
print("Mock dataset OK.")