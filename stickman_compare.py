import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import sys
from omegaconf import OmegaConf
from os.path import join as pjoin

# 引入專案依賴
from mGPT.archs.mgpt_vq import VQVae
from visualize.feature_to_joints_v4 import recover_g1_motion_v4

# ================= 🔧 核心配置區 =================
# 1. 目標編號 (6位數字)
TARGET_NAME = "000064"

# 2. SMPL 基準資料路徑 (Side A)
SMPL_JOINTS_DIR = "datasets/humanml3d/new_joints"

# 3. G1 Token 轉換設定 (Side B)
G1_TOKEN_DIR = "datasets/humanml3d/TOKENS"
CKPT_PATH_G1 = "experiments/mgpt/VQVAE_UnitreeG1/checkpoints/epoch=1099-v1.ckpt" 
CONFIG_PATH = "configs/config_ug1_stage1.yaml"
VQ_CONFIG_PATH = "configs/vq/default.yaml"
MEAN_STD_PATH_G1 = "datasets/unitreeg1/meta"
NFEATS = 363 
# ===============================================

# ================= 🦴 骨架拓扑定义 =================
SMPL_SKELETON_EDGES = [
    # 左腿 (Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Foot)
    (0, 1), (1, 4), (4, 7), (7, 10),
    # 右腿 (Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Foot)
    (0, 2), (2, 5), (5, 8), (8, 11),
    # 躯干与头 (Pelvis -> Spine1 -> Spine2 -> Spine3 -> Neck -> Head)
    (0, 3), (3, 6), (6, 9), (9, 12), (12, 15),
    # 左臂 (Spine3 -> L_Collar -> L_Shoulder -> L_Elbow -> L_Wrist)
    (9, 13), (13, 16), (16, 18), (18, 20),
    # 右臂 (Spine3 -> R_Collar -> R_Shoulder -> R_Elbow -> R_Wrist)
    (9, 14), (14, 17), (17, 19), (19, 21)
]

G1_SKELETON_EDGES = [
    (0,1), (1,2), (2,3), (3,4), (4,5), (5,6), 
    (0,7), (7,8), (8,9), (9,10), (10,11), (11,12), 
    (0,13), (13,14), (14,15), 
    (15,16), (16,17), (17,18), (18,19), (19,20), (20,21), (21,22), 
    (15,23), (23,24), (24,25), (25,26), (26,27), (27,28), 
]
# ===============================================

def load_g1_vae_model(device):
    """加載 G1 的 VQ-VAE 解碼器，並強制過濾 SMPL 凍結權重"""
    base_cfg = OmegaConf.load(CONFIG_PATH)
    vq_cfg = OmegaConf.load(VQ_CONFIG_PATH)
    
    vae_params = vq_cfg.params
    vae_params.nfeats = NFEATS
    vae_params.ablation = base_cfg.LOSS.ABLATION 
    
    model = VQVae(**vae_params).to(device)
    
    try:
        checkpoint = torch.load(CKPT_PATH_G1, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(CKPT_PATH_G1, map_location=device)

    state_dict = checkpoint.get('state_dict', checkpoint)
    new_state_dict = {}
    for k, v in state_dict.items():
        if 'smpl_' in k: continue # 物理阻斷污染
        if k.startswith('motion_vae.'): new_state_dict[k.replace('motion_vae.', '', 1)] = v
        elif k.startswith('vae.'): new_state_dict[k.replace('vae.', '', 1)] = v
        elif k.startswith('encoder.') or k.startswith('decoder.') or k.startswith('quantizer.'): new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

def get_full_sequence_data(device):
    """執行全序列讀取與推演，返回記憶體緩存的 numpy 陣列"""
    smpl_path = pjoin(SMPL_JOINTS_DIR, TARGET_NAME + ".npy")
    token_path = pjoin(G1_TOKEN_DIR, TARGET_NAME + ".npy")
    
    if not os.path.exists(smpl_path):
        raise FileNotFoundError(f"缺失 SMPL 基準資料: {smpl_path}")
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"缺失 G1 Token 資料: {token_path}")

    # 1. 處理 SMPL (Side A)
    smpl_joints = np.load(smpl_path) # [T, 22, 3]
    
    # 2. 處理 G1 (Side B)
    model_g1 = load_g1_vae_model(device)
    t_mean_g1 = torch.tensor(np.load(pjoin(MEAN_STD_PATH_G1, "Mean.npy")), dtype=torch.float32, device=device)
    t_std_g1 = torch.tensor(np.load(pjoin(MEAN_STD_PATH_G1, "Std.npy")), dtype=torch.float32, device=device)
    
    tokens = np.load(token_path)
    if len(tokens.shape) == 1:
        tokens = np.expand_dims(tokens, axis=0)
    token_tensor = torch.from_numpy(tokens).long().to(device)
    
    with torch.no_grad():
        feat_recon_norm = model_g1.decode(token_tensor)
        feat_recon_tensor = feat_recon_norm * t_std_g1 + t_mean_g1
        pos_recon_tensor, _ = recover_g1_motion_v4(feat_recon_tensor)
    
    g1_joints = pos_recon_tensor[0].cpu().numpy() # [T, 30, 3]

    return smpl_joints, g1_joints

def init_axes(ax, title, current_root):
    """統一視角初始化"""
    radius = 1.0
    ax.set_xlim3d([current_root[0] - radius, current_root[0] + radius])
    ax.set_ylim3d([current_root[2] - radius, current_root[2] + radius])
    ax.set_zlim3d([0, 1.5])
    ax.set_xlabel('X (Left/Right)')
    ax.set_ylabel('Z (Forward/Back)')
    ax.set_zlabel('Y (Height)')
    ax.set_title(title)
    ax.view_init(elev=15, azim=45)
    ax.set_aspect('equal')

def draw_skeleton(ax, pos, edges, is_g1=False):
    """繪製單幀骨架，進行 XY-Z 軸向映射轉換"""
    for i, (j1, j2) in enumerate(edges):
        if j1 < pos.shape[0] and j2 < pos.shape[0]:
            if is_g1:
                # 沿用原 G1 的色彩邏輯
                c = 'r' if i in [0,1,2,3,4,5] else 'b' if i in [6,7,8,9,10,11] else 'k'
            else:
                c = 'b' # SMPL 全局使用藍色作為對比
                
            # X->X, Y(高度)->Z, Z(深度)->Y 映射至 Matplotlib 3D 坐標系
            ax.plot([pos[j1,0], pos[j2,0]], [pos[j1,2], pos[j2,2]], [pos[j1,1], pos[j2,1]], c=c, lw=2)
            
    ax.scatter(pos[0,0], pos[0,2], pos[0,1], c='r', marker='s', s=50) # 標記 Root

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"啟動跨實體驗證程序 (目標編號: {TARGET_NAME})")
    print("正在執行全序列特徵重構與逆運動學推演...")
    
    smpl_joints, g1_joints = get_full_sequence_data(device)
    
    T = min(smpl_joints.shape[0], g1_joints.shape[0])
    print(f"資料對齊完畢，有效比對幀數: {T} 幀。")
    
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    fig.suptitle(f"Cross-Embodiment Alignment Check: {TARGET_NAME}", fontsize=16)
    
    def update(frame):
        ax1.cla()
        init_axes(ax1, f"Side A: SMPL Reference (Frame {frame})", smpl_joints[frame, 0])
        draw_skeleton(ax1, smpl_joints[frame], SMPL_SKELETON_EDGES, is_g1=False)

        ax2.cla()
        init_axes(ax2, f"Side B: G1 Token Reconstruction (Frame {frame})", g1_joints[frame, 0])
        draw_skeleton(ax2, g1_joints[frame], G1_SKELETON_EDGES, is_g1=True)

    ani = FuncAnimation(fig, update, frames=T, interval=50)
    plt.show()

if __name__ == "__main__":
    main()