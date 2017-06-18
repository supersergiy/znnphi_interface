#!/bin/bash
./data_creator.py
./net_creator.py
./reference_run.py
tar -czf bundle.tar.gz nets data

