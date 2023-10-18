#!/bin/bash

torchrun --standalone --nnodes=1 --nproc-per-node=2 ddp_dataset.py
