#!/usr/bin/env python
#
# A simple script to call the mrtrix.sh shell script on many subjects in parallel.
#
# @author Bob Dougherty

import subprocess
import shlex
import os
import sys
from glob import glob
import time
import random

from threading import Thread

sub_dirs = glob('/hcp/*')
# Run subjects in random order.
random.seed()
random.shuffle(sub_dirs)
#sub_dirs = sorted(glob('/data/hcp/data/*'))

# Set the number of parallel jobs. Note that the mrtrix tools are multi-threaded, so
# you should keep this fairly low to avoid too much contention. I've found that running
# a few jobs (e.g., 4) in parallel is net faster on a 32-core system than just relying
# on the parallelism in the mrtrix tools.
num_jobs = 4

# Use a queue for the output of each command so we can get some reasonable output
# in the terminal to track progress.
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty # for Python 3.x

def enqueue_output(out, queue):
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

processes = []
q = Queue()
threads = []
num_running = 1
while num_running>0 or len(sub_dirs)>0:
    num_running = sum([p.poll()==None for p in processes])
    if num_running < num_jobs and len(sub_dirs)>0:
        raw_dir = sub_dirs.pop(0)
        sub_code = os.path.basename(raw_dir)
        sub_dir = os.path.join('/data','hcp','data',sub_code)
        if os.path.exists(os.path.join(sub_dir,'2M_SIFT.tck')) or os.path.exists(os.path.join(sub_dir,'2M_SIFT.trk')):
            print('Subject %s is already done-- skipping...' % sub_code)
        elif os.path.exists(os.path.join(raw_dir,'T1w','Diffusion')):
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            cmd = './mrtrix.sh %s %s' % (os.path.join('/hcp',sub_code,'T1w/Diffusion'), sub_dir)
            print(cmd)
            p = subprocess.Popen(args=shlex.split(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1)
            processes.append(p)
            t = Thread(target=enqueue_output, args=(p.stdout, q))
            threads.append(t)
            t.daemon = True
            t.start()
            time.sleep(0.5)
    try:
        line = q.get_nowait()
    except Empty:
        pass
    else:
        sys.stdout.write(line)
    time.sleep(0.2)

print 'All processes done'



