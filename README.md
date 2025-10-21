# Tensorboard set-up test

## 1. Find your log directory

The logs are being written to:

```bash
/users/1/lundq163/projects/automated-qc/src/training/runs/<tb_prefix>/<time_str>-trn_cls-<comment>/
/users/1/lundq163/projects/automated-qc/src/training/runs/<tb_prefix>/<time_str>-val_cls-<comment>/
```

## 2. Launch TensorBoard on the cluster

SSH into your cluster and run:

```bash
cd /users/1/lundq163/projects/automated-qc/src/training
/users/1/lundq163/projects/automated-qc/.venv/bin/tensorboard --logdir=runs --port=6006 --bind_all
```

## 3. Create an SSH tunnel from your local machine

On your local machine, open a terminal and run:

```bash
ssh -L 6006:localhost:6006 lundq163@<your-cluster-hostname>
```

Replace `<your-cluster-hostname>` with your actual cluster address (e.g., `login.msi.umn.edu` or similar for UMN systems).

## 4. Open TensorBoard in your browser

Navigate to:

<http://localhost:6006>

You should now see your training metrics updating in real-time!

Tips:

- Keep the SSH tunnel open while you want to monitor training

- If port 6006 is already in use, try a different port (e.g., --port=6007)

- The --bind_all flag allows TensorBoard to be accessible from any network interface

Alternative if you're on MSI (UMN): MSI may have a web portal or specific instructions for port forwarding. Check their documentation or use their OnDemand portal if available.
