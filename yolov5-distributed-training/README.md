# Distributed Training with YOLOv5

A guide to training a [YOLOv5](https://github.com/ultralytics/yolov5) model across two lab PCs (one **master**, one **worker**) using PyTorch distributed training over SSH.

## Overview

| Role | Description |
|---|---|
| **Master** | Coordinates training (`node_rank 0`) and hosts the rendezvous address and port |
| **Worker** | Joins the master's training run (`node_rank 1`) |

Both machines have one GPU each (`--nproc_per_node 1`), for two nodes in total.

## Table of contents

1. [Prerequisites](#prerequisites)
2. [Pre-setup: network, SSH and hostnames](#1-pre-setup-network-ssh-and-hostnames)
3. [Set up YOLOv5](#2-set-up-yolov5-for-distributed-training)
4. [Start training](#3-start-training)
5. [Troubleshooting notes](#notes-and-troubleshooting)

## Prerequisites

- Two Ubuntu PCs on the same network (examples below use `192.168.25.XXX`)
- `sudo` access on both machines
- Python and `pip` installed
- A labelled dataset on [Roboflow](https://roboflow.com)

Throughout this guide, `LABPCXX` means the PC's name (e.g. `LABPC11`) and `XXX` means the last part of that machine's IP address. Run the setup in section 1 on **both** machines.

## 1. Pre-setup: network, SSH and hostnames

### Set a static IP address

1. Open the network settings and set a static IPv4 address. Keep the other settings as shown, but replace the address with the one you want for the machine (`192.168.25.XXX`).

   ![Static IPv4 settings](images/static-ip-settings.png)

2. Turn the network off and on again so the change applies.

   ![Toggle the wired connection](images/network-toggle.png)

### Install and start OpenSSH (one time)

3. In a terminal:

   ```bash
   sudo apt-get install openssh-server
   sudo systemctl start ssh
   ```

   To restart the SSH server later:

   ```bash
   sudo systemctl restart ssh
   ```

### Set the hostname

4. Move to the `/etc` directory:

   ```bash
   cd /etc
   ```

5. Edit the hosts file and change the hostname to `LABPCXX` (not case-sensitive):

   ```bash
   sudo nano hosts
   ```

   ![Editing /etc/hosts](images/etc-hosts.png)

6. Edit the hostname file and change it to `LABPCXX` as well:

   ```bash
   sudo nano hostname
   ```

   ![Editing /etc/hostname](images/etc-hostname.png)

7. Leave the `/etc` folder:

   ```bash
   cd
   ```

### Set up SSH keys

Generate a key pair. This may need to be repeated after a reboot.

```bash
ssh-keygen -t rsa -b 4096 -C "LABPCXX"   # replace XX with the PC number
```

Press Enter at every prompt. The output should look like this:

![ssh-keygen output](images/ssh-keygen-output.png)

Then copy the key to the other machine:

```bash
ssh-copy-id labuser@192.168.25.179   # replace with the other PC's IP
```

**Do this on both PCs** so the master and the worker can each reach the other.

## 2. Set up YOLOv5 for distributed training

1. Clone YOLOv5 and install its requirements (on both machines, or on the master and copy as in step 4):

   ```bash
   git clone https://github.com/ultralytics/yolov5   # clone
   cd yolov5
   pip install -r requirements.txt
   ```

2. Export your dataset from Roboflow using the **YOLO v5 PyTorch** format and choose *download zip to computer*.

   ![Roboflow export dialog](images/roboflow-export.png)

3. Extract the zip and put the `train`, `valid` and `test` folders and the `data.yaml` file into the `yolov5` folder.

4. Copy the project from the master to the worker:

   ```bash
   scp -r /home/labuser/yolov5 labuser@192.168.25.XXX:/home/labuser   # XXX = end of the worker's IP
   ```

## 3. Start training

In a terminal, go to the `yolov5` folder on each machine and run the matching command.

### On the master

```bash
python -m torch.distributed.run \
  --nproc_per_node 1 \
  --nnodes 2 \
  --node_rank 0 \
  --master_addr "192.168.25.XXX" \
  --master_port 1234 \
  train.py \
  --batch 32 \
  --data data.yaml \
  --cfg yolov5s.yaml \
  --weights ' '
```

### On the worker

```bash
python -m torch.distributed.run \
  --nproc_per_node 1 \
  --nnodes 2 \
  --node_rank 1 \
  --master_addr "192.168.1.1" \
  --master_port 1234 \
  train.py \
  --batch 64 \
  --data data.yaml \
  --cfg yolov5s.yaml \
  --weights ' '
```

### What the flags mean

| Flag | Meaning |
|---|---|
| `--nproc_per_node 1` | One training process (one GPU) per machine |
| `--nnodes 2` | Total number of machines |
| `--node_rank` | `0` for the master, `1` for the worker |
| `--master_addr` | IP address of the master machine |
| `--master_port` | Port used for coordination (must match on both) |
| `--batch` | Batch size |
| `--data data.yaml` | Dataset config exported from Roboflow |
| `--cfg yolov5s.yaml` | Model architecture (YOLOv5 small) |
| `--weights ' '` | Empty value, so training starts from scratch instead of pretrained weights |

## Notes and troubleshooting

- **Worker `--master_addr`:** The original notes show `192.168.1.1` on the worker, but the master's address is on the `192.168.25.XXX` network. Both commands should normally use the **master's** IP, so check this value before running.
- **Batch size:** The notes use `--batch 32` on the master and `--batch 64` on the worker. YOLOv5 treats `--batch` as the total batch size across all processes, so the two nodes are normally given the same value. If training hangs or errors at startup, try using the same number on both.
- **Start order:** Start the master first, then the worker. The master waits until all nodes have joined.
- **SSH keys:** If `ssh` or `scp` asks for a password again after a reboot, re-run the key setup.
- **Same environment:** Both machines need matching Python packages. Run `pip install -r requirements.txt` on the worker too.
