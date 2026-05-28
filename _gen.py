import os
os.makedirs(r"F:\VQVAE\utils", exist_ok=True)
os.makedirs(r"F:\VQVAE\models", exist_ok=True)

# Write a simple test
with open(r"F:\VQVAE\utils\physics_losses.py", "w", encoding="utf-8") as f:
    f.write("# test\nprint('hello')\n")
print("OK")
