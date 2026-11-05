"""Conditional GPU GEMV prototype; branches precede all weight reads/MACs."""
import mlx.core as mx

class ConditionalMLP:
    def __init__(self):
        self.up=mx.fast.metal_kernel(name='dlropt_gate_up',input_names=['x','wg','wu','reuse'],output_names=['middle','mac_branches'],source='''
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        if (reuse[0]) {
            if (lane == 0) { middle[row] = T(0); if (row == 0) mac_branches[0] = 0; }
            return;
        }
        float g = 0.0f, u = 0.0f;
        for (uint col = lane; col < D; col += 32) {
            float a = float(x[col]);
            g += a * float(wg[row * D + col]);
            u += a * float(wu[row * D + col]);
        }
        g = simd_sum(g); u = simd_sum(u);
        if (lane == 0) {
            float bg = float(T(g)), bu = float(T(u));
            middle[row] = T((bg / (1.0f + metal::exp(-bg))) * bu);
            if (row == 0) mac_branches[0] = 2;
        }
        ''')
        self.down=mx.fast.metal_kernel(name='dlropt_down',input_names=['middle','wd','cached','reuse'],output_names=['out','mac_branches'],source='''
        uint row = threadgroup_position_in_grid.x;
        uint lane = thread_position_in_threadgroup.x;
        if (reuse[0]) {
            if (lane == 0) { out[row] = cached[row]; if (row == 0) mac_branches[0] = 0; }
            return;
        }
        float value = 0.0f;
        for (uint col = lane; col < I; col += 32)
            value += float(middle[col]) * float(wd[row * I + col]);
        value = simd_sum(value);
        if (lane == 0) { out[row] = T(value); if (row == 0) mac_branches[0] = 1; }
        ''')

    def __call__(self,mlp,x,cached,reuse):
        d=x.shape[-1];i=mlp.gate_proj.weight.shape[0];reuse=reuse.reshape(1)
        middle,up_count=self.up(inputs=[x,mlp.gate_proj.weight,mlp.up_proj.weight,reuse],template=[('T',x.dtype),('D',d)],grid=(i*32,1,1),threadgroup=(32,1,1),output_shapes=[(i,),(1,)],output_dtypes=[x.dtype,mx.uint32])
        out,down_count=self.down(inputs=[middle,mlp.down_proj.weight,cached,reuse],template=[('T',x.dtype),('I',i)],grid=(d*32,1,1),threadgroup=(32,1,1),output_shapes=[x.shape,(1,)],output_dtypes=[x.dtype,mx.uint32])
        return out,up_count+down_count
