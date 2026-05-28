with open(r"F:\VQVAE\train_vq_v3.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix: args.activation -> getattr for compatibility
content = content.replace(
    "activation=args.activation, norm=args.norm,",
    "activation=getattr(args, \"activation\", getattr(args, \"vq_act\", \"relu\")), norm=getattr(args, \"norm\", getattr(args, \"vq_norm\", None)),"
)

with open(r"F:\VQVAE\train_vq_v3.py", "w", encoding="utf-8") as f:
    f.write(content)
print("Fixed")
