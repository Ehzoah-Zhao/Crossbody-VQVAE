import os

def w(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

# physics_losses.py
w(r"F:\VQVAE\utils\physics_losses.py", open(r"F:\VQVAE\utils\physics_losses.py","r").read() if os.path.exists(r"F:\VQVAE\utils\physics_losses.py") else "")

print("test")
