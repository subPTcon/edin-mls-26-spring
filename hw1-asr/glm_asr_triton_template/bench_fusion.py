import torch                                                                                                      
import time     
from layers import Linear, MLP, EncoderMLP, gelu, silu                                                            
                                                                                                                
device = "cuda"                                                                                                   
Linear.BACKEND = "triton"                                                                                         
                                                                                                                
seq_lens = [1, 64, 256, 512, 1024, 2048, 4096]                                                                                
                                                                                                                
# ---- EncoderMLP (Linear+GELU) ----                                                                              
hidden = 1280   
intermediate = 5120                                                                                               
encoder_mlp = EncoderMLP(hidden, intermediate, activation="gelu")                                                 
encoder_mlp.fc1.weight = encoder_mlp.fc1.weight.to(device)                                                        
encoder_mlp.fc2.weight = encoder_mlp.fc2.weight.to(device)                                                        
if encoder_mlp.fc1.bias_param is not None:                                                                        
    encoder_mlp.fc1.bias_param = encoder_mlp.fc1.bias_param.to(device)                                            
if encoder_mlp.fc2.bias_param is not None:                                                                        
    encoder_mlp.fc2.bias_param = encoder_mlp.fc2.bias_param.to(device)                                            
                                                                                                                
print("=== EncoderMLP (Linear+GELU) ===")                                                                         
print(f"{'M':>6}  {'Fused':>10}  {'Unfused':>10}  {'Speedup':>8}")                                                
for seq in seq_lens:                                                                                              
    x = torch.randn(1, seq, hidden, device=device, dtype=torch.float32)                                           
    # warmup both paths                                                                                           
    EncoderMLP.FUSED = True; encoder_mlp._fc1_weight_t = None                                                     
    for _ in range(3): _ = encoder_mlp(x)                                                                         
    EncoderMLP.FUSED = False                                                                                      
    for _ in range(3): _ = encoder_mlp(x)                                                                         
    torch.cuda.synchronize()                                                                                      
                                                                                                                
    # Fused                                                                                                       
    EncoderMLP.FUSED = True; encoder_mlp._fc1_weight_t = None                                                     
    times_f = []                                                                                                  
    for _ in range(20):                                                                                           
        torch.cuda.synchronize(); t0 = time.perf_counter()                                                        
        _ = encoder_mlp(x)                                                                                        
        torch.cuda.synchronize(); t1 = time.perf_counter()                                                        
        times_f.append((t1 - t0) * 1000)                                                                          
                                                                                                                
    # Unfused                                                                                                     
    EncoderMLP.FUSED = False                                                                                      
    times_u = []                                                                                                  
    for _ in range(20):                                                                                           
        torch.cuda.synchronize(); t0 = time.perf_counter()                                                        
        _ = encoder_mlp(x)                                                                                        
        torch.cuda.synchronize(); t1 = time.perf_counter()                                                        
        times_u.append((t1 - t0) * 1000)                                                                          
                                                                                                                
    f = sum(times_f)/len(times_f)                                                                                 
    u = sum(times_u)/len(times_u)                                                                                 
    print(f"{seq:>6}  {f:>9.3f}ms  {u:>9.3f}ms  {u/f:>7.2f}x")                                                    
                                                                                                                
# ---- MLP SwiGLU ----                                                                                            
hidden_d = 3584                                                                                                   
intermediate_d = 18944                                                                                            
mlp = MLP(hidden_d, intermediate_d, activation="silu", use_gating=True)                                           
mlp.gate_proj.weight = mlp.gate_proj.weight.to(device)                                                            
mlp.up_proj.weight = mlp.up_proj.weight.to(device)                                                                
mlp.down_proj.weight = mlp.down_proj.weight.to(device)                                                            
                                                                                                                
print("\n=== MLP SwiGLU (Linear+SiLU+Gate) ===")                                                                  
print(f"{'M':>6}  {'Fused':>10}  {'Unfused':>10}  {'Speedup':>8}")                                                
for seq in seq_lens:                                                                                              
    x_d = torch.randn(1, seq, hidden_d, device=device, dtype=torch.float32)                                       
    MLP.FUSED = True; mlp._gate_weight_t = None; mlp._up_weight_t = None                                          
    for _ in range(3): _ = mlp(x_d)                                                                               
    MLP.FUSED = False                                                                                             
    for _ in range(3): _ = mlp(x_d)                                                                               
    torch.cuda.synchronize()                                                                                      
                                                                                                                
    MLP.FUSED = True; mlp._gate_weight_t = None; mlp._up_weight_t = None                                          
    times_f = []                                                                                                  
    for _ in range(20):                                                                                           
        torch.cuda.synchronize(); t0 = time.perf_counter()                                                        
        _ = mlp(x_d)                                                                                              
        torch.cuda.synchronize(); t1 = time.perf_counter()                                                        
        times_f.append((t1 - t0) * 1000)                                                                          
                                                                                                                
    MLP.FUSED = False                                                                                             
    times_u = []                                                                                                  
    for _ in range(20):                                                                                           
        torch.cuda.synchronize(); t0 = time.perf_counter()                                                        
        _ = mlp(x_d)                                                                                              
        torch.cuda.synchronize(); t1 = time.perf_counter()                                                        
        times_u.append((t1 - t0) * 1000)                                                                          
                                                                                                                
    f = sum(times_f)/len(times_f)                                                                                 
    u = sum(times_u)/len(times_u)                                                                                 
    print(f"{seq:>6}  {f:>9.3f}ms  {u:>9.3f}ms  {u/f:>7.2f}x")