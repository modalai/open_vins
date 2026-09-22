#!/usr/bin/env python3
"""Independent 70-digit dense reference for the archived first EKF update.

Requires numpy and mpmath only. Run --write-reference to regenerate the final
reference block; without it, verify the checked-in binary reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import mpmath as mp
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('--write-reference',action='store_true')
args=parser.parse_args()
path=Path(__file__).with_name('ekf_update_183x86.bin')
data=path.read_bytes();offset=16
assert data[:16]==b'OV_EKF_SNAPSHOT1'
n,m,hc,nv,no=struct.unpack_from('<5Q',data,offset);offset+=40
def matrix(rows,cols):
 global offset
 a=np.frombuffer(data,dtype='<f8',count=rows*cols,offset=offset).reshape((rows,cols),order='F').copy();offset+=8*rows*cols
 return a
P=matrix(n,n);H=matrix(m,hc);r=matrix(m,1);R=matrix(m,m)
variables={}
for _ in range(nv):
 i,size,rows=struct.unpack_from('<3i',data,offset);offset+=12
 variables[i]=(size,matrix(rows,1),matrix(rows,1))
order=struct.unpack_from('<'+'i'*no,data,offset);offset+=4*no
manifest=json.loads(path.with_suffix('.json').read_text())
assert offset==manifest['input_bytes']
assert hashlib.sha256(data[:offset]).hexdigest()==manifest['input_sha256']
Hd=np.zeros((m,n));column=0
for i in order:
 size=variables[i][0];Hd[:,i:i+size]=H[:,column:column+size];column+=size
assert column==hc
mp.mp.dps=70
def high(a):return mp.matrix([[mp.mpf(float(v)) for v in row] for row in a])
Ph=high((P+P.T)/2);Hh=high(Hd);Rh=high(R);rh=high(r)
M=Ph*Hh.T;S=Hh*M+Rh;K=M*(S**-1)
reference_P=np.array((Ph-K*M.T).tolist(),dtype='<f8')
reference_dx=np.array((K*rh).tolist(),dtype='<f8')
reference=reference_P.tobytes(order='F')+reference_dx.tobytes(order='F')
if args.write_reference:
 path.write_bytes(data[:offset]+reference)
 manifest['reference_sha256']=hashlib.sha256(reference).hexdigest()
 manifest['binary_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
 manifest['prior_max_asymmetry']=float(np.max(np.abs(P-P.T)))
 path.with_suffix('.json').write_text(json.dumps(manifest,indent=2)+'\n')
else:
 assert data[offset:]==reference, 'reference differs from the 70-digit independent computation'
print(json.dumps({'result':'PASS','digits':mp.mp.dps,'input_sha256':manifest['input_sha256'],
 'reference_sha256':hashlib.sha256(reference).hexdigest(),'prior_max_asymmetry':float(np.max(np.abs(P-P.T)))}))
