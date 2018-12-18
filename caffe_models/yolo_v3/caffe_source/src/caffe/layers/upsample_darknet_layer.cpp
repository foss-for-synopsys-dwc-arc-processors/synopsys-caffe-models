
 #include <vector>  
 #include "caffe/layers/upsample_darknet_layer.hpp"  
   
 namespace caffe {  
   
 template <typename Dtype>  
 void UpsampleDarknetLayer<Dtype>::LayerSetUp(  
   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
   UpsampleDarknetParameter upsample_darknet_param = this->layer_param_.upsample_darknet_param();  
   stride_ = upsample_darknet_param.stride();  
 }  
   
 template <typename Dtype>  
 void UpsampleDarknetLayer<Dtype>::Reshape(  
   const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {  
   vector<int> out_shape;  
   for (int i = 0; i < bottom[0]->num_axes(); i++) {  
     out_shape.push_back(bottom[0]->shape(i));  
   }  
   
   out_shape[bottom[0]->num_axes() - 1] *= stride_;  
   out_shape[bottom[0]->num_axes() - 2] *= stride_;  
   top[0]->Reshape(out_shape);  
 }  
   
 template <typename Dtype>  
 void UpsampleDarknetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,  
     const vector<Blob<Dtype>*>& top) {  
   
   int N = top[0]->shape(0);  
   int C = top[0]->shape(1);  
   int H = top[0]->shape(2);  
   int W = top[0]->shape(3);  
   
   const Dtype *input = bottom[0]->cpu_data();  
   Dtype *output = top[0]->mutable_cpu_data();  
   for (int n = 0; n < N; n++) {  
     for (int c = 0; c < C; c++) {  
       for (int h = 0; h < H; h++) {  
         for (int w = 0; w < W; w++) {  
           int nw = w/stride_;  
           int nh = h/stride_;  
           int out_idx = (((n * C + c) * H) + h) * W + w;  
           int in_idx = (((n * C + c) * (H / stride_)) + nh) * (W / stride_) + nw;  
           output[out_idx] = input[in_idx];  
         }  
       }  
     }  
   }  
 }  
   
 template <typename Dtype>  
 void UpsampleDarknetLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,  
     const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {  
   int N = bottom[0]->shape(0);  
   int C = bottom[0]->shape(1);  
   int H = bottom[0]->shape(2);  
   int W = bottom[0]->shape(3);  
   const Dtype *output_grad = top[0]->cpu_diff();  
   Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();  
   caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);  
   for (int n = 0; n < N; n++) {  
     for (int c = 0; c < C; c++) {  
       for (int h = 0; h < H; h++) {  
         for (int w = 0; w < W; w++) {  
           for (int i = 0; i < stride_; i++) {  
             for (int j = 0; j < stride_; j++) {  
               int nw = w * stride_ + i;  
               int nh = h * stride_ + j;  
               int out_idx = (((n * C + c) * H) + h) * W + w;  
               int in_idx = (((n * C + c) * (H * stride_))  
                   + nh) * (W * stride_) + nw;  
               bottom_diff[out_idx] += output_grad[in_idx];  
             }  
           }  
         }  
       }  
     }  
   }  
 }  
   
 #ifdef CPU_ONLY  
 STUB_GPU(UpsampleDarknetLayer);  
 #endif  
   
 INSTANTIATE_CLASS(UpsampleDarknetLayer);  
 REGISTER_LAYER_CLASS(UpsampleDarknet);  
   
 }  // namespace caffe  
   

