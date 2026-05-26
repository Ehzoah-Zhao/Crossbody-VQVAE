import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Cross-Embodiment VQVAE Training Config',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## ── 1. 数据集路径与基础配置 (请根据本地实际路径修改 default) ──────────────────
    parser.add_argument('--data_dir_A', type=str, default='./dataset/smpl/new_joint_vecs', 
                        help='SMPL 动作数据目录 (剔除了Location，仅保留Rotation等)')
    parser.add_argument('--data_dir_B', type=str, default='./dataset/unitreeg1/new_joint_vecs', 
                        help='Unitree G1 动作数据目录 (剔除了Location，仅保留Rotation等)')
    parser.add_argument('--split_txt_dir', type=str, default='./dataset/splits',
                        help='存放 train.txt, val.txt, test.txt 的目录')
    parser.add_argument('--stat_dir_A', type=str, default='./dataset/smpl/meta',
                        help='存放 SMPL mean.npy 和 std.npy 的目录')
    parser.add_argument('--stat_dir_B', type=str, default='./dataset/unitreeg1/meta',
                        help='存放 G1 mean.npy 和 std.npy 的目录')

    ## ── 2. 本体维度配置 ──────────────────────────────────────────────────
    parser.add_argument('--input_dim_A', type=int, default=134,
                        help='SMPL 输入特征维度: Root(4)+Rot(21*6)+Contact(4)')
    parser.add_argument('--input_dim_B', type=int, default=186,
                        help='G1 输入特征维度:  Root(4)+Rot(30*6)+Contact(2)')

    ## ── 3. 训练常规超参数 ──────────────────────────────────────────────────
    parser.add_argument('--batch-size', default=256, type=int, help='独立训练时的 batch size')
    parser.add_argument('--batch_size_paired', type=int, default=128, help='配对数据对比损失的 batch size')
    parser.add_argument('--window-size', type=int, default=64, help='训练切片固定的动作帧数长')
    parser.add_argument('--total-iter', default=200000, type=int, help='总迭代步数')
    parser.add_argument('--warm-up-iter', default=5000, type=int, help='Warm-up 阶段步数')
    parser.add_argument('--lr', default=2e-4, type=float, help='最大学习率')
    parser.add_argument('--lr-scheduler', default=[50000, 120000], nargs="+", type=int, help="学习率衰减节点步数")
    parser.add_argument('--gamma', default=0.05, type=float, help="学习率衰减率")
    parser.add_argument('--weight-decay', default=0.0, type=float, help='权重衰减')

    ## ── 4. VQVAE 核心架构超参 ───────────────────────────────────────────────
    parser.add_argument("--code-dim", type=int, default=512, help="Embedding 码本特征维度")
    parser.add_argument("--nb-code", type=int, default=512, help="Codebook Token 总数量")
    parser.add_argument("--mu", type=float, default=0.99, help="EMA 更新量化器的动量超参")
    parser.add_argument("--down-t", type=int, default=2, help="时间维度下采样率次数")
    parser.add_argument("--stride-t", type=int, default=2, help="卷积步长")
    parser.add_argument("--width", type=int, default=512, help="网络各层通道宽度")
    parser.add_argument("--depth", type=int, default=3, help="ResNet 块深度")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="膨胀卷积增长率")
    parser.add_argument("--output-emb-width", type=int, default=512, help="输出层特征宽度")
    parser.add_argument('--vq-act', type=str, default='relu', choices=['relu', 'silu', 'gelu'], help='激活函数类型')
    parser.add_argument('--vq-norm', type=str, default=None, help='归一化层类型(None/LayerNorm/BatchNorm)')
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices=['ema', 'orig', 'ema_reset', 'reset'], help="量化器更新策略")

    ## ── 5. 核心多任务 Loss 权重 (对应课程学习三大阶段) ──────────────────────────
    parser.add_argument('--recons-loss', type=str, default='l1_smooth', choices=['l1', 'l2', 'l1_smooth'], help='重建损失函数类型')
    parser.add_argument('--w_recon', type=float, default=1.0, help='自重建损失基础权重')
    parser.add_argument('--w_commit', type=float, default=0.02, help='VQ 承诺损失权重')
    parser.add_argument('--w_vel', type=float, default=0.1, help='速度/帧差分平滑损失权重')
    
    # 动态调参项 (运行脚本或在此直接改默认值)
    parser.add_argument('--w_cycle', type=float, default=0.0, 
                        help='循环一致性损失权重 (Warm-up阶段设为0.0，Main早期设为0.05，后期拉高至0.1)')
    parser.add_argument('--w_contrast', type=float, default=0.0, 
                        help='配对对比损失权重 (有配对数据参与训练时，后期拉高至0.05)')
    parser.add_argument('--temperature', type=float, default=0.07, help='InfoNCE 对比损失中的温度系数')

    ## ── 6. 实验输出与保存 ──────────────────────────────────────────────────
    parser.add_argument('--out-dir', type=str, default='output_vqfinal/', help='模型与日志总输出根目录')
    parser.add_argument('--exp-name', type=str, default='exp_cross_embodiment', help='当前实验子文件夹名称')
    parser.add_argument('--print-iter', default=200, type=int, help='日志打印步数频率')
    parser.add_argument('--save_iter', type=int, default=10000, help='模型权重保存步数频率')
    parser.add_argument('--seed', default=123, type=int, help='随机种子')
    parser.add_argument("--resume-pth", type=str, default=None, help='恢复训练的 checkpoint 权重路径 (用于断点续传)')

    return parser.parse_args()