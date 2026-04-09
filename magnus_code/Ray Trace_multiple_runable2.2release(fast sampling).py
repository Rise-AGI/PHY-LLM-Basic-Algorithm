import numpy as np
import cupy as cp
import cv2
import time
from cupy.cuda import texture
from cupy.cuda import runtime
print('import succeed.')
nr=1.5168
ng=1.52
nb=1.5168
class sphericalLens:
    def __init__(self, r1, r2, z, d, rm):
        self.r1 = float(r1)
        self.r2 = float(r2)
        self.d = float(d)
        self.rm = float(rm)
        self.signr1 = float(np.sign(r1))
        self.signr2 = float(np.sign(r2))
        self.z = float(z)
        self.zc1 = float(-d/2 + np.sign(r1)*np.sqrt(r1**2 - rm**2))
        self.zc2 = float(d/2 - np.sign(r2)*np.sqrt(r2**2 - rm**2))


#透镜
l1 = sphericalLens(r1=10, r2=10, z=25, d=0.2, rm=1.1) 
lenses=[l1]
numlenses =len(lenses)
max_lenses = 1#必须同步在c源码里面同步修改这个数，这里写一个报错防住，这个数字能小则小，小了能够节省4-6个ms的运行时间


# 屏的参数
w, h = 1000, 500
pw, ph = 20.0, 10.0
numm = 512
total_threads = w * h * 3 


#像的参数
z_Object = 11      
resx,resy = 512,512
z_step = 0.1             
sizex,sizey = 10,10





if numlenses>max_lenses: raise ValueError('透镜数量过多，请同步修改此处代码的max_lenses和下面CUDA C源码里面的MAX_LENSES才能通过')
r1_arr=cp.asarray([l.r1 for l in lenses],dtype=np.float32)
r2_arr=cp.asarray([l.r2 for l in lenses],dtype=np.float32)
z_arr=cp.asarray([l.z for l in lenses],dtype=np.float32)
d_arr=cp.asarray([l.d for l in lenses],dtype=np.float32)
rm_arr=cp.asarray([l.rm for l in lenses],dtype=np.float32)
sgnr1_arr=cp.asarray([l.signr1 for l in lenses],dtype=np.float32)
sgnr2_arr=cp.asarray([l.signr2 for l in lenses],dtype=np.float32)
zc1_arr=cp.asarray([l.zc1 for l in lenses],dtype=np.float32)
zc2_arr=cp.asarray([l.zc2 for l in lenses],dtype=np.float32)
n_arr=cp.asarray([nr,ng,nb],dtype=np.float32)

cuda_source_code='''
__device__ unsigned int hash_rng(unsigned int seed) {
    seed ^= seed >> 16;
    seed *= 0x7feb352dU;
    seed ^= seed >> 15;
    seed *= 0x846ca68bU;
    seed ^= seed >> 16;
    return seed;
}

// 产生 [0.0, 1.0) 的标准均匀分布浮点数
__device__ float rand_float(unsigned int& seed) {
    seed = hash_rng(seed);
    return (float)seed * 2.3283064365386963e-10f; 
}


extern "C" __global__
void render_kernel(
float* __restrict__ final_img,
cudaTextureObject_t tex_obj,
const float* __restrict__ r1_arr,
const float* __restrict__ r2_arr,
const float* __restrict__ z_arr,
const float* __restrict__ d_arr,
const float* __restrict__ rm_arr,
const float* __restrict__ sgnr1_arr,
const float* __restrict__ sgnr2_arr,
const float* __restrict__ zc1_arr,
const float* __restrict__ zc2_arr,
const float* __restrict__ n_arr,
float z_obj,float sizex,float sizey,int resx,int resy,int w,int h,float pw,float ph,int numm,int total_threads,int numlenses,int frames
){
    
    //在这里修改MAX_LENSES
    
    const int MAX_LENSES = 1;





    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;
    
    int pixel_idx = tid / 3; 
    int px = pixel_idx % w;     
    int py = pixel_idx / w;
    int colorid = tid % 3;

    float n = __ldg(&n_arr[colorid]);
    float inv_w = 1.0f / (float)w;
    float inv_h = 1.0f / (float)h;
    float inv_sizex = 1.0f / sizex;
    float inv_sizey = 1.0f / sizey;
    float nn = n * n;
    float nninv = 1.0f / nn;

    float phys_x = ((float)px * inv_w - 0.5f) * pw;
    float phys_y = (0.5f - (float)py * inv_h) * ph;

    float rm1 = __ldg(&rm_arr[0]);
    float rmsq = rm1*rm1;
    float initial_z_lens = __ldg(&z_arr[0]);

    float color_sum = 0.0f;
    float4 pixel;
    float tx;float ty;float t_obj;float fx;float fy;
    unsigned int seed = hash_rng(pixel_idx + 1+hash_rng(frames));
float l_zc1[MAX_LENSES], l_zc2[MAX_LENSES], l_r1[MAX_LENSES], l_r2[MAX_LENSES], l_rm[MAX_LENSES], l_s1[MAX_LENSES], l_s2[MAX_LENSES], l_z[MAX_LENSES];
for(int i=0; i < numlenses && i < MAX_LENSES; ++i){
    l_zc1[i] = __ldg(&zc1_arr[i]);
    l_zc2[i] = __ldg(&zc2_arr[i]);
    l_r1[i] = __ldg(&r1_arr[i]);
    l_r2[i] = __ldg(&r2_arr[i]);
    l_rm[i] = __ldg(&rm_arr[i]);
    l_s1[i] = __ldg(&sgnr1_arr[i]);
    l_s2[i] = __ldg(&sgnr2_arr[i]);
    l_z[i] = __ldg(&z_arr[i]);
}
    for(int m = 0; m < numm; ++m){
        float u = rand_float(seed);
        float v = rand_float(seed);

        float r_lens = rm1 * sqrtf(u);
        float th = 6.28318530718f * v; 
        float sin_th, cos_th;
        sincosf(th, &sin_th, &cos_th);
        float lx = r_lens * cos_th;
        float ly = r_lens * sin_th;

        float dx = lx - phys_x;
        float dy = ly - phys_y;
        float dz = initial_z_lens;  

        float inv_d_norm = rsqrtf(dx*dx + dy*dy + dz*dz + 1e-10f);
        dx *= inv_d_norm; dy *= inv_d_norm; dz *= inv_d_norm;
        
        float current_z_lens = initial_z_lens;
        float ox, oy, oz;

        for (int lensid = 0; lensid < numlenses; ++lensid){
            float zc1 = l_zc1[lensid];
            float zc2 = l_zc2[lensid];
            float r1 = l_r1[lensid];
            float r2 = l_r2[lensid];
            float r1sq = r1 * r1;
            float r2sq = r2 * r2;
            float signr1 = l_s1[lensid];
            float signr2 = l_s2[lensid];
            float rm = l_rm[lensid];
            rmsq = rm * rm;

            ox = (lensid == 0) ? phys_x : ox;
            oy = (lensid == 0) ? phys_y : oy;
            oz = -current_z_lens;
            float ocz = oz - zc1;
            
            float b = ox * dx + oy * dy + ocz * dz;
            float c = ox * ox + oy * oy + ocz * ocz - r1sq;

            float delta = b * b - c;
            if ( delta < 0.0f ) goto next_ray;

            float t_hit1 = -b - signr1 * sqrtf(delta);
            float x1 = ox + t_hit1 * dx;
            float y1 = oy + t_hit1 * dy;
            float z1 = oz + t_hit1 * dz;
            if ((x1 * x1 + y1 * y1) > rmsq) goto next_ray;
            if ((x1*x1+y1*y1)>1.0f) goto next_ray;

            float temp1 = z1 - zc1;
            float inv_n_norm = rsqrtf(x1 * x1 + y1 * y1 + temp1 * temp1);
            float nvx = x1 * inv_n_norm;
            float nvy = y1 * inv_n_norm;
            float nvz = temp1 * inv_n_norm;
            float nv = nvx * dx + nvy * dy + nvz * dz;
            if (nv<0.0f){ nvx=-nvx; nvy=-nvy; nvz=-nvz; nv=-nv; }
            
            float x = nv * nv + nn - 1.0f;
            if ( x < 0.0f ) goto next_ray;
            float k = -nv + sqrtf(x);
            float d1x = dx + k * nvx;
            float d1y = dy + k * nvy;
            float d1z = dz + k * nvz;
            float inv_d1_norm = rsqrtf(d1x * d1x + d1y * d1y + d1z * d1z + 1e-10f);
            d1x *= inv_d1_norm; d1y *= inv_d1_norm; d1z *= inv_d1_norm;

            float ocz2 = z1 - zc2;
            float bb = x1 * d1x + y1 * d1y + ocz2 * d1z;
            float cc = x1 * x1 + y1 * y1 + ocz2 * ocz2 - r2sq;
            float ddelta = bb * bb - cc;
            if (ddelta<0.0f) goto next_ray;
            
            float t_hit2 = -bb + signr2 * sqrtf(ddelta);
            float x2 = x1 + t_hit2 * d1x;
            float y2 = y1 + t_hit2 * d1y;
            float z2 = z1 + t_hit2 * d1z;

            float n2x = x2;
            float n2y = y2;
            float n2z = z2 - zc2;
            float inv_n2_norm = rsqrtf(n2x*n2x + n2y*n2y + n2z*n2z + 1e-10f);
            n2x *= inv_n2_norm; n2y *= inv_n2_norm; n2z *= inv_n2_norm;

            float nv2 = n2x * d1x + n2y * d1y + n2z * d1z;
            if (nv2 < 0.0f) { n2x = -n2x; n2y = -n2y; n2z = -n2z; nv2 = -nv2; }
            float tmp2 = nv2 * nv2 + nninv - 1.0f;
            if (tmp2 < 0.0f) goto next_ray; 

            float k2 = -nv2 + sqrtf(tmp2);
            float d2x = d1x + k2 * n2x;
            float d2y = d1y + k2 * n2y;
            float d2z = d1z + k2 * n2z;
            
            float inv_d2_norm = rsqrtf(d2x*d2x + d2y*d2y + d2z*d2z + 1e-10f);
            d2x *= inv_d2_norm; d2y *= inv_d2_norm; d2z *= inv_d2_norm;

            ox = x2;
            oy = y2;
            dx = d2x;
            dy = d2y;
            dz = d2z;

            if (lensid + 1 < numlenses) {
                float zzz = __ldg(&z_arr[lensid+1]);
                current_z_lens = zzz - z2;
            } else {
                current_z_lens = z2;
            }
        }

        t_obj = (z_obj - current_z_lens) / dz;
        if (t_obj <= 0.0f) goto next_ray;

        fx = ox + t_obj * dx;
        fy = oy + t_obj * dy;

        tx = (fx * inv_sizex + 0.5f) * (float)resx;
        ty = (0.5f - fy * inv_sizey) * (float)resy;
if (tx>0 && ty>0 && tx<(float)resx && ty<(float)resy){
        pixel = tex2D<float4>(tex_obj, tx, ty);
        if (colorid==0) color_sum+=pixel.x;
        else if (colorid==1) color_sum+=pixel.y;
        else color_sum+=pixel.z;}

    next_ray:;
    }
    float avg_color = color_sum / (float)numm;
    //float gamma_corrected = __powf(avg_color, 0.454545f);
    //float scaled = gamma_corrected * 255.0f;
    //scaled = fmaxf(0.0f, fminf(scaled, 255.0f));
    //final_img[tid] = (unsigned char)scaled;
    final_img[tid] = avg_color;
}
'''
render_fused_kernel = cp.RawKernel(cuda_source_code, 'render_kernel',options=('-use_fast_math',))
print('defs ended')
# img_cpu = np.zeros((resy, resx, 3), dtype=np.float32)
# cv2.putText(img_cpu, "Meow!", (100, 276), cv2.FONT_HERSHEY_SIMPLEX, 2, (0.0, 1.0, 1.0), 5)


def generate_grid_object(resy, resx):
    # 创建黑色背景 (float32, RGB)
    grid = np.zeros((resy, resx, 3), dtype=np.float32)
    
    step = 32  # 网格间距
    thickness = 2 # 线条粗细

    # 1. 绘制细分网格线 (青色)
    for i in range(0, resx, step):
        cv2.line(grid, (i, 0), (i, resy), (0, 0.8, 0.8), thickness)
    for j in range(0, resy, step):
        cv2.line(grid, (0, j), (resx, j), (0, 0.8, 0.8), thickness)

    # 2. 绘制主轴 (增强辨识度)
    # X轴中心线 (绿色)
    cv2.line(grid, (0, resy // 2), (resx, resy // 2), (0, 1.0, 0), thickness + 2)
    # Y轴中心线 (红色)
    cv2.line(grid, (resx // 2, 0), (resx // 2, resy), (1.0, 0, 0), thickness + 2)

    # 3. 在中心画几个同心圆 (观察球差和爱因斯坦环非常有用)
    center = (resx // 2, resy // 2)
    for r in [64, 128, 192]:
        cv2.circle(grid, center, r, (1.0, 1.0, 1.0), thickness)

    return grid

# --- 在你的主程序中替换 ---
img_cpu = generate_grid_object(resy, resx)


img = cp.asarray(img_cpu)
def create_texture_object(img_cp):
    h, w, c = img_cp.shape
    bytes_per_pixel = 16 
    alignment = 256
    pitch_bytes = ((w * bytes_per_pixel + alignment - 1) // alignment) * alignment
    padded_w = pitch_bytes // bytes_per_pixel
    rgba = cp.zeros((h, padded_w, 4), dtype=cp.float32)
    rgba[:, :w, :3] = img_cp
    ch_fmt = texture.ChannelFormatDescriptor(32, 32, 32, 32, runtime.cudaChannelFormatKindFloat)
    res_ptr = texture.ResourceDescriptor(
        runtime.cudaResourceTypePitch2D, 
        arr=rgba,                  
        chDesc=ch_fmt, 
        width=w,
        height=h,
        pitchInBytes=pitch_bytes
    )
    tex_ptr = texture.TextureDescriptor(
        addressModes=(runtime.cudaAddressModeClamp, runtime.cudaAddressModeBorder),borderColors = (0.0,0.0,0.0,0.0),
        filterMode=runtime.cudaFilterModeLinear,
        readMode=runtime.cudaReadModeElementType
    )
    tex_obj = texture.TextureObject(res_ptr, tex_ptr)
    return tex_obj, rgba
tex_handle, _internal_storage = create_texture_object(img)
final_img = cp.zeros((w * h * 3), dtype=cp.float32)
threads_per_block = 256
blocks_per_grid = (total_threads + threads_per_block - 1) // threads_per_block
cv2.namedWindow("Lens Rendering")
cv2.namedWindow("Object")
img_cpu = cv2.cvtColor(img_cpu, cv2.COLOR_RGB2BGR)
cv2.imshow("Object", img_cpu*255)
running = True
frames=0
# 1. 准备缓冲区 (float32)
accumulated_buffer = cp.zeros((h * w * 3), dtype=cp.float32)
# 准备当前帧的临时存放区
current_frame_float = cp.zeros((h * w * 3), dtype=cp.float32)
print("Controls: [W] move forward, [ESC] quit")
while running:
    
    t0 = time.time()
    render_fused_kernel(
        (blocks_per_grid,), (threads_per_block,),
        (current_frame_float, tex_handle.ptr,
         r1_arr,r2_arr,z_arr,d_arr,rm_arr,sgnr1_arr,sgnr2_arr,zc1_arr,zc2_arr,n_arr,np.float32(z_Object),np.float32(sizex),np.float32(sizey),np.int32(resx),np.int32(resy),np.int32(w),np.int32(h),
         np.float32(pw),np.float32(ph),np.int32(numm),np.int32(total_threads),np.int32(numlenses),np.int32(frames))
    )
    # cp.cuda.Stream.null.synchronize()
    key = cv2.waitKey(1) & 0xFF
    if key == ord('w'):
        z_Object += z_step
        accumulated_buffer.fill(0)
        frames = 0
        print(f"Moving forward → z_Object = {z_Object:.1f}")
    elif key == ord('W'):
        z_Object += 5*z_step
        accumulated_buffer.fill(0)
        frames = 0
        print(f"Moving forward → z_Object = {z_Object:.1f}")
    elif key == ord('s'):
        z_Object -= z_step
        accumulated_buffer.fill(0)
        frames = 0
        print(f"Moving backward → z_Object = {z_Object:.1f}")
    elif key == ord('S'):
        z_Object -= 5*z_step
        accumulated_buffer.fill(0)
        frames = 0
        print(f"Moving forward → z_Object = {z_Object:.1f}")
    elif key == 27: 
        running = False
        print("Exiting...")
    accumulated_buffer += current_frame_float
    frames+=1
    avg_img = accumulated_buffer / frames

    display_img = cp.clip(cp.power(avg_img, 0.4545) * 255.0, 0, 255).astype(cp.uint8)

    render_img = cp.asnumpy(display_img).reshape((h, w, 3))
    render_img = cv2.cvtColor(render_img, cv2.COLOR_RGB2BGR)
    cv2.imshow("Lens Rendering", render_img)
    t1 = time.time()
    print(f'Render time (z={z_Object:.1f}): {1/(t1-t0+0.0001):.4f}FPS')
    

cv2.destroyAllWindows()