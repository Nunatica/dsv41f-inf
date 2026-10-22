#!/usr/bin/env python3
"""Small deterministic V4.1 graph fixture; tests only, never a serving model."""
import ctypes as C
import argparse
import json
from pathlib import Path
import struct
import sys
import numpy as np
root=Path(__file__).resolve().parents[1]
lib=C.CDLL(str(root/'build/bin/libggml-base.so'))
lib.ggml_quantize_chunk.argtypes=[C.c_int,C.c_void_p,C.c_void_p,C.c_longlong,C.c_longlong,C.c_longlong,C.c_void_p]
lib.ggml_quantize_chunk.restype=C.c_size_t
rng=np.random.default_rng(41)
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('output',nargs='?',type=Path,default=Path('/tmp/dsv4f-v41-fixture.gguf'))
parser.add_argument('--mtp',action='store_true',help='also create a small DSpark raw binary/JSON sidecar')
parser.add_argument('--indexer-heads',type=int,choices=[2,4,32,64],default=2)
parser.add_argument('--indexer-topk',type=int,default=8)
args=parser.parse_args()
if args.indexer_topk<1:parser.error('--indexer-topk must be positive')
D,FF,DH,NH,RQ,OG,OL,IH,ID,NE,V=256,256,128,2,128,2,128,args.indexer_heads,128,6,32
meta={'general.architecture':'deepseek41','general.name':'dsv4f-v41-test-fixture',
'deepseek41.hidden_size':D,'deepseek41.num_hidden_layers':3,'deepseek41.num_attention_heads':NH,'deepseek41.head_dim':DH,'deepseek41.qk_rope_head_dim':64,'deepseek41.q_lora_rank':RQ,'deepseek41.o_lora_rank':OL,'deepseek41.o_groups':OG,'deepseek41.sliding_window':8,'deepseek41.n_routed_experts':NE,'deepseek41.num_experts_per_tok':3,'deepseek41.moe_intermediate_size':FF,'deepseek41.n_shared_experts':1,'deepseek41.hc_mult':4,'deepseek41.hc_sinkhorn_iters':4,'deepseek41.index_n_heads':IH,'deepseek41.index_head_dim':ID,'deepseek41.index_topk':args.indexer_topk,'deepseek41.candidate_source_layer_id':1,'deepseek41.candidate_topk_blocks':8,'deepseek41.candidate_block_size':8,'deepseek41.vocab_size':V,'deepseek41.max_position_embeddings':128,'deepseek41.routed_scaling_factor':.1,'deepseek41.hc_eps':1e-6,'deepseek41.rms_norm_eps':1e-6,'deepseek41.rope_theta':10000,'deepseek41.compress_rope_theta':1000000,'deepseek41.rope_scaling.factor':1.,'deepseek41.rope_scaling.beta_fast':32.,'deepseek41.rope_scaling.beta_slow':1.,'deepseek41.rope_scaling.original_max_position_embeddings':128.,'deepseek41.swiglu_limit':1.25,'deepseek41.norm_topk_prob':True,'deepseek41.scoring_func':'sqrtsoftplus','deepseek41.compress_ratios':[0,1,2],'deepseek41.kv_source_layer_ids':[1,2],'deepseek41.index_source_layer_ids':[1,2],
'deepseek41.config':json.dumps({'engram_n_heads':2,'engram_head_dim':32,'engram_max_ngram_size':2}),'deepseek41.engram.pad_id':2,'deepseek41.engram.compressed_vocab_size':V,'deepseek41.engram.layer_ids':[1],'deepseek41.engram.rows':[18],'deepseek41.engram.token_map':[0xffffffff if i==10 else i for i in range(V)],'deepseek41.engram.primes':[7,11],'deepseek41.engram.multipliers':[11,13],
 'tokenizer.ggml.model':'gpt2','tokenizer.ggml.pre':'joyai-llm','tokenizer.ggml.tokens':['<bos>','<eos>']+[f'tok{i}' for i in range(2,V)],'tokenizer.ggml.token_type':[3,3]+[1]*(V-2),'tokenizer.ggml.merges':[],'tokenizer.ggml.bos_token_id':0,'tokenizer.ggml.eos_token_id':1,'tokenizer.ggml.add_bos_token':False}
if args.mtp:
 # Remove routed activation-quantization noise from the target parity fixture.
 meta['deepseek41.routed_scaling_factor']=0.0
 config=json.loads(meta['deepseek41.config'])
 config.update(dspark_block_size=5,dspark_noise_token_id=3,dspark_target_layer_ids=[0,1,2],
               dspark_n_routed_experts=4,dspark_num_experts_per_tok=2,dspark_markov_rank=32)
 meta['deepseek41.config']=json.dumps(config)
tensors=[]
geom={0:(1,4),1:(1,2),8:(32,34),11:(256,110),42:(64,18),143:(128,28),26:(1,4)}
def add(name,shape,ty=0,fill=None):
 # Dense F32 isolates graph/cache semantics from MMVQ vs MMQ activation rounding.
 if ty in (1,8):ty=0
 n=int(np.prod(shape));be,bb=geom[ty]
 if ty==26:data=np.asarray(fill,dtype='<i4').tobytes()
 elif ty==143:
  b=rng.integers(0,256,(n//128,28),dtype=np.uint8);scale=np.float16(.025).tobytes();b[:,26]=scale[0];b[:,27]=scale[1];data=b.tobytes()
 else:
  a=(rng.normal(0,.04,n) if fill is None else np.broadcast_to(fill,(n,))).astype(np.float32)
  if ty==0:data=a.astype('<f4').tobytes()
  elif ty==1:data=a.astype('<f2').tobytes()
  else:
   buf=C.create_string_buffer(n//be*bb);wrote=lib.ggml_quantize_chunk(ty,a.ctypes.data,C.addressof(buf),0,n//shape[0],shape[0],None);assert wrote==len(buf);data=buf.raw
 tensors.append((name,shape,ty,data))
add('token_embd.weight',[D,V],1);add('output_norm.weight',[D],fill=1);add('output.weight',[D,V],8)
for l in range(3):
 b=f'blk.{l}.'
 for site in ['attn','ffn']:
  add(b+f'hc_{site}_fn.weight',[D*4,24],1);add(b+f'hc_{site}_base.weight',[24],fill=0);add(b+f'hc_{site}_scale.weight',[3],fill=1);add(b+site+'_norm.weight',[D],fill=1)
 for name,shape,ty,fill in [('attn_sinks.weight',[NH],0,0),('attn_q_a.weight',[D,RQ],8,None),('attn_q_b.weight',[RQ,NH*DH],8,None),('attn_q_a_norm.weight',[RQ],0,1),('attn_kv.weight',[D,DH],8,None),('attn_kv_a_norm.weight',[DH],0,1),('attn_output_a.weight',[NH*DH//OG,OG*OL],8,None),('attn_output_b.weight',[OG*OL,D],8,None),('ffn_gate_inp.weight',[D,NE],0,None),('exp_probs_b.bias',[NE],0,np.arange(NE,dtype=np.float32)),('exp_probs_b_vl.bias',[NE],0,0),('ffn_gate_shexp.weight',[D,FF],8,None),('ffn_up_shexp.weight',[D,FF],8,None),('ffn_down_shexp.weight',[FF,D],8,None)]:add(b+name,shape,ty,fill)
 if l==1:
  add(b+'engram_kv.weight',[64,D*5]);add(b+'engram_q_norm.weight',[D,4],fill=1);add(b+'engram_k_norm.weight',[D,4],fill=1)
 if l:
  add(b+'attn_compressor_kv.weight',[D,DH],1);add(b+'attn_compressor_norm.weight',[DH],fill=1);add(b+'indexer.attn_k.weight',[DH,ID],1);add(b+'indexer.k_norm.weight',[ID],fill=1);add(b+'indexer.attn_q_b.weight',[RQ,IH*ID],1);add(b+'indexer.proj.weight',[D,IH],1)
  if l==2:add(b+'attn_compressor_gate.weight',[D,DH],1)
 for mapname in ['gu','down']:add(b+f'ffn_exps_{mapname}_type',[NE],26,[11,42,143,11,42,143])
 for ty,suffix in [(11,'q3_k'),(42,'q2_0'),(143,'ptq1_0')]:
  for role,shape in [('gate',[D,FF,2]),('up',[D,FF,2]),('down',[FF,D,2])]:add(b+f'ffn_{role}_exps.{suffix}.weight',shape,ty)
def string(v):
 s=v.encode();return struct.pack('<Q',len(s))+s
def kv(k,v):
 if isinstance(v,bool):ty,data=7,struct.pack('<?',v)
 elif isinstance(v,int):ty,data=4,struct.pack('<I',v)
 elif isinstance(v,float):ty,data=6,struct.pack('<f',v)
 elif isinstance(v,str):ty,data=8,string(v)
 else:
  ty=9;et=8 if k.endswith(('tokens','merges')) else 10 if k.endswith('multipliers') else 5 if k.endswith('token_type') else 4
  data=struct.pack('<IQ',et,len(v))+b''.join(string(x) if et==8 else struct.pack('<Q' if et==10 else '<i' if et==5 else '<I',x) for x in v)
 return string(k)+struct.pack('<I',ty)+data
head=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(meta))+b''.join(kv(k,v) for k,v in meta.items());payload=bytearray()
for name,shape,ty,data in tensors:
 head+=string(name)+struct.pack('<I',len(shape))+struct.pack('<'+'Q'*len(shape),*shape)+struct.pack('<IQ',ty,len(payload));payload+=data;payload+=bytes(-len(payload)%32)
head+=bytes(-len(head)%32)
path=args.output;path.write_bytes(head+payload);print(path,len(head+payload))

engram_dir=path.with_suffix('.engram');engram_dir.mkdir(exist_ok=True)
w=(np.arange(18*32,dtype=np.uint16)%110).astype(np.uint8).tobytes();sc=bytes([121]*18)
h=json.dumps({'layers.1.engram.embed.weight':{'dtype':'F8_E4M3','shape':[18,32],'data_offsets':[0,len(w)]},'layers.1.engram.embed.scale':{'dtype':'F8_E8M0','shape':[18,1],'data_offsets':[len(w),len(w)+len(sc)]}}).encode()
(engram_dir/'fixture.safetensors').write_bytes(struct.pack('<Q',len(h))+h+w+sc)
print('DSV4F_ENGRAM_DIR='+str(engram_dir))

if args.mtp:
 # Mirror the real manifest: PyTorch shapes, separate experts, BF16 and Q8_0.
 index=[];payload=bytearray()
 def mtp_add(name,shape,typ='Q8_0',fill=None):
  n=int(np.prod(shape));a=(rng.normal(0,.02,n) if fill is None else np.broadcast_to(fill,(n,))).astype(np.float32)
  if typ=='F32':data=a.astype('<f4').tobytes()
  elif typ=='BF16':data=(a.view(np.uint32)>>16).astype('<u2').tobytes()
  else:
   ty,bs,bb={'Q8_0':(8,32,34),'IQ2_XXS':(16,256,66)}[typ]
   buf=C.create_string_buffer(n//bs*bb);importance=np.ones(shape[0],dtype=np.float32)
   wrote=lib.ggml_quantize_chunk(ty,a.ctypes.data,C.addressof(buf),0,n//shape[0],shape[0],importance.ctypes.data)
   assert wrote==len(buf);data=buf.raw
  index.append([name,typ,len(payload),len(data),list(reversed(shape))])
  payload.extend(data);payload.extend(bytes(-len(payload)%256))
 for b in range(3):
  prefix=f'mtp.{b}.'
  for name,shape in [('attn.wkv.weight',[D,DH]),('attn.wo_a.weight',[NH*DH//OG,OG*OL]),('attn.wo_b.weight',[OG*OL,D]),
                     ('attn.wq_a.weight',[D,RQ]),('attn.wq_b.weight',[RQ,NH*DH]),
                     ('ffn.shared_experts.w1.weight',[D,FF]),('ffn.shared_experts.w3.weight',[D,FF]),('ffn.shared_experts.w2.weight',[FF,D])]:mtp_add(prefix+name,shape)
  for name,shape in [('attn.kv_norm.weight',[DH]),('attn.q_norm.weight',[RQ]),('attn_norm.weight',[D]),('ffn_norm.weight',[D])]:mtp_add(prefix+name,shape,'BF16',1)
  mtp_add(prefix+'attn.attn_sink',[NH],'F32',0)
  mtp_add(prefix+'ffn.gate.bias',[4],'F32',0);mtp_add(prefix+'ffn.gate.bias_vl',[4],'F32',0)
  mtp_add(prefix+'ffn.gate.weight',[D,4],'BF16')
  for site in ['attn','ffn']:
   mtp_add(prefix+f'hc_{site}_base',[24],'F32',0);mtp_add(prefix+f'hc_{site}_scale',[3],'F32',1);mtp_add(prefix+f'hc_{site}_fn',[4*D,24],'F32')
  for e in range(4):
   for w,shape in [('w1',[D,FF]),('w3',[D,FF]),('w2',[FF,D])]:mtp_add(prefix+f'ffn.experts.{e}.{w}.weight',shape,'IQ2_XXS')
 mtp_add('mtp.0.main_proj.weight',[D*3,D]);mtp_add('mtp.0.main_norm.weight',[D],'BF16',1);mtp_add('mtp.2.norm.weight',[D],'BF16',1)
 mtp_add('mtp.2.markov_head.embed.weight',[32,V],'BF16');mtp_add('mtp.2.markov_head.head.weight',[32,V],'BF16')
 mtp_add('mtp.2.confidence_head.proj.weight',[D+32,1],'BF16')
 # A deliberately impossible vision payload proves it is never read/allocated.
 index.append(['vision.do_not_load.weight','F32',1<<40,1<<30,[16384,16384]])
 binary=path.with_suffix('.mtp.bin');manifest=path.with_suffix('.mtp.json')
 binary.write_bytes(payload);manifest.write_text(json.dumps({'tensors':index}))
 print(binary,manifest)
