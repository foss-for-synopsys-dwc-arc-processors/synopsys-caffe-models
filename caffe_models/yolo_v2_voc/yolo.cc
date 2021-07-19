#include <math.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PRINTLINE (printf("LINE %d\n",__LINE__), fflush(stdout));

typedef float Result;

// Detection layer parmameers.
namespace Region_parms {
    static const int num=5;
    static const int classes=20;
    static const int coords=4;
    // Not specified in the prototxt, but hard-coded.  See src/detector.c from darknet.
    static const float thresh = 0.5;
    static const float nms = 0.4;
    // Here are the biases from yolo-voc.cfg.
    static Result anchors[] = { 1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071};
    }
struct Box {
    float x, y, w, h;
    };
typedef float Prob;

struct Layer {
    int batch;
    int n;
    int h,w;
    float *biases;
    Result *output;
    int classes;
    int inputs,outputs;
    Result *delta;
    float *cost, *bias_updates;
    int coords;
    };

static inline float logistic_activate(float x) {return 1./(1. + exp(-x));}

Box get_region_box(
        float *x, float *biases, int n, int index, int i, int j, int w, int h) {
    Box b;
    static int bnum = 0;
    if (0 && bnum++ < 5) {
        printf("grb i %d j %d x %5.2f %5.2f %5.2f %5.2f bias %5.2f\n",
                i,j,x[index],x[index+1],x[index+2],x[index+3],biases[2*n]);
        }
    b.x = (i + logistic_activate(x[index + 0])) / w;
    b.y = (j + logistic_activate(x[index + 1])) / h;
    b.w = exp(x[index + 2]) * biases[2*n]   / w;
    b.h = exp(x[index + 3]) * biases[2*n+1] / h;
    return b;
    }

void get_region_boxes(
        Layer l, int w, int h, float thresh, 
        float **probs, Box *boxes, int only_objectness, int *map, float tree_thresh) {
    int i,j,n;
    float *predictions = l.output;
    const int trace = 0;
    trace && printf("get region boxes w %d h %d l.w %d l.h %d t1 %f t2 %f\n",
        w,h,l.w,l.h,thresh, tree_thresh);
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int index = i*l.n + n;
            int p_index = index * (l.classes + 5) + 4;
            float scale = predictions[p_index];
            int box_index = index * (l.classes + 5);
            boxes[index] = get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
            boxes[index].x *= w;
            boxes[index].y *= h;
            boxes[index].w *= w;
            boxes[index].h *= h;
            trace && printf("box %d %5.2f %5.2f %5.2f %5.2f scale %5.2f\n",index,
                boxes[index].x, boxes[index].y, boxes[index].w, boxes[index].h, scale);

            int class_index = index * (l.classes + 5) + 5;
            for (j = 0; j < l.classes; ++j) {
                float prob = scale*predictions[class_index+j];
                if (trace && prob > thresh) 
                    printf("i %d j %d prob %5.2f ci+j %d scale %f pred %f\n",
                        i,j,prob,class_index+j,
                        scale,predictions[class_index+j]);
                if (prob > 1) { printf("HEY!  \n"); }
                probs[index][j] = (prob > thresh) ? prob : 0;
                }
            if (only_objectness) {
                probs[index][0] = scale;
                }
            }
        }
    }

struct Sortable_bbox {
    int index;
    int klass;
    Prob **probs;
    };

int nms_comparator(const void *pa, const void *pb) {
    Sortable_bbox a = *(Sortable_bbox *)pa;
    Sortable_bbox b = *(Sortable_bbox *)pb;
    Prob diff = a.probs[a.index][b.klass] - b.probs[b.index][b.klass];
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
    }

float overlap(float x1, float w1, float x2, float w2) {
    float l1 = x1 - w1/2;
    float l2 = x2 - w2/2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1/2;
    float r2 = x2 + w2/2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
    }

float box_intersection(Box a, Box b) {
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0) return 0;
    float area = w*h;
    return area;
    }

float box_union(Box a, Box b) {
    float i = box_intersection(a, b);
    float u = a.w*a.h + b.w*b.h - i;
    return u;
    }

float box_iou(Box a, Box b) {
    return box_intersection(a, b)/box_union(a, b);
    }

void do_nms_sort(Box *boxes, Prob **probs, int total, int classes, float thresh) {
    Sortable_bbox *s = new Sortable_bbox[total];

    for (int i = 0; i < total; i++) {
        s[i].index = i;       
        s[i].klass = 0;
        s[i].probs = probs;
        }

    for (int k = 0; k < classes; k++) {
        for (int i = 0; i < total; i++) {
            s[i].klass = k;
            }
        qsort(s, total, sizeof(Sortable_bbox), nms_comparator);
        for (int i = 0; i < total; i++) {
            if (probs[s[i].index][k] == 0) continue;
            Box a = boxes[s[i].index];
            for (int j = i+1; j < total; j++) {
                Box b = boxes[s[j].index];
                if (box_iou(a, b) > thresh) {
                    probs[s[j].index][k] = 0;
                    }
                }
            }
        }
    delete s;
    }

#if 1
static char *voc_names[] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"};
#else
static char * voc_names[] = {
    "person", "bicycle", "car", "motorbike", "aeroplane", 
    "bus", "train", "truck", "boat", "trafficlight", 
    "firehydrant", "stopsign", "parkingmeter", "bench", "bird", 
    "cat", "dog", "horse", "sheep", "cow", 
    "elephant", "bear", "zebra", "giraffe", "backpack", 
    "umbrella", "handbag", "tie", "suitcase", "frisbee", 
    "skis", "snowboard", "sportsball", "kite", "baseballbat", 
    "baseballglove", "skateboard", "surfboard", "tennisracket", "bottle", 
    "wineglass", "cup", "fork", "knife", "spoon", 
    "bowl", "banana", "apple", "sandwich", "orange", 
    "broccoli", "carrot", "hotdog", "pizza", "donut", 
    "cake", "chair", "sofa", "pottedplant", "bed", 
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", 
    "remote", "keyboard", "cellphone", "microwave", "oven", 
    "toaster", "sink", "refrigerator", "book", "clock", 
    "vase", "scissors", "teddybear", "hairdrier", "toothbrush",  };
#endif

int max_index(Prob *a, int n) {
    if (n <= 0) return -1;
    int i, max_i = 0;
    Prob max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max) {
            max = a[i];
            max_i = i;
            }
        }
    return max_i;
    }

float get_color(int c, int x, int max) {
    static float colors[6][3] = { {1,0,1}, {0,0,1},{0,1,1},{0,1,0},{1,1,0},{1,0,0} };
    float ratio = ((float)x/max)*5;
    int i = floor(ratio);
    int j = ceil(ratio);
    ratio -= i;
    float r = (1-ratio) * colors[i][c] + ratio*colors[j][c];
    //printf("%f\n", r);
    return r;
    }

struct Image {
    short h,w;
    };

void draw_box_width(
        Image a, int x1, int y1, int x2, int y2, int w, float r, float g, float b) {
    printf("    draw box %d %d %d %d %d\n", x1, y1, x2, y2, w);
    }

void draw_detections(
        Image im, int num, float thresh, Box *boxes, Prob **probs, 
        char **names, Image **alphabet, int classes) {

    for (int i = 0; i < num; ++i) {
        int klass = max_index(probs[i], classes);
        Prob prob = probs[i][klass];
        if (prob > thresh) {

            int width = im.h * .012;

            printf("%s: %.0f%%\n", names[klass], prob*100);
            int offset = klass*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            Box b = boxes[i];

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if (left < 0) left = 0;
            if (right > im.w-1) right = im.w-1;
            if (top < 0) top = 0;
            if (bot > im.h-1) bot = im.h-1;

            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            #if 0
            if (alphabet) {
                Image label = get_label(alphabet, names[klass], (im.h*.03)/10);
                draw_label(im, top + width, left, label, rgb);
                }
            #endif
            }
        }
    }

Layer make_region_layer(int batch, int w, int h, int n, int classes, int coords) {
    Layer L;
    L.batch = batch;
    L.n = n;
    L.h = h;
    L.w = w;
    L.coords = coords;
    L.classes = classes;
    L.cost = new Result; L.cost = 0;
    // From make_region_layer in region_layer.c
    L.biases = new Result[n*2];
    for (int i = 0; i < n*2; i++) L.biases[i] = .5;
    memcpy(L.biases,Region_parms::anchors,sizeof(Region_parms::anchors));
    L.bias_updates = (Result*)calloc(n*2,sizeof(Result));
    L.inputs = L.outputs = h*w*n*(classes + coords + 1);
    // Used in forward by darknet.  We can probably delete.
    L.delta = (Result*)calloc(batch*L.outputs, sizeof(float)); 
    L.output = (Result*)calloc(batch*L.outputs, sizeof(float));
    return L;
    }

void flatten(float *x, int size, int layers, int batch, int forward) {
    float *swap = (float*)calloc(size*layers*batch, sizeof(float));
    int i,c,b;
    for(b = 0; b < batch; ++b){
        for(c = 0; c < layers; ++c){
            for(i = 0; i < size; ++i){
                int i1 = b*layers*size + c*size + i;
                int i2 = b*layers*size + i*layers + c;
                if (forward) swap[i2] = x[i1];
                else swap[i1] = x[i2];
                }
            }
        }
    memcpy(x, swap, size*layers*batch*sizeof(float));
    free(swap);
    }

void softmax(float *input, int n, float temp, float *output) {
    int i;
    float sum = 0;
    float largest = -FLT_MAX;
    for(i = 0; i < n; ++i){
        if(input[i] > largest) largest = input[i];
        }
    for(i = 0; i < n; ++i){
        float e = exp(input[i]/temp - largest/temp);
        sum += e;
        output[i] = e;
        }
    for(i = 0; i < n; ++i){
        output[i] /= sum;
        }
    }
void forward_region_layer(const Layer l, Result *fresult) {
    int i,j,b,t,n;
    // size*n is the number of input maps.  E.g. (4 + 80 + 1 = size 85) x n = 5 => 425
    int size = l.coords + l.classes + 1;
    printf("forward of region layer n=%d %dx%d n=%d size=%d outputs=%d\n",
        l.n, l.w,l.h,l.n,size,l.outputs);
    memcpy(l.output, fresult, l.outputs*l.batch*sizeof(Result));
    const int SHOW_INPUT = 0;
    if (SHOW_INPUT) {
        printf("region input\n");
        for (int i = 0; i < l.outputs; i++) {
            printf("[%4d] = %5.4f\n",i,fresult[i]);
            }
        }

    flatten(l.output, l.w*l.h, size*l.n, l.batch, 1);
    if (SHOW_INPUT) {
        printf("region after flatten\n");
        for (int i = 0; i < l.outputs; i++) {
            printf("[%4d] = %5.4f\n",i,l.output[i]);
            }
        }
    // softmax=1 in yolo.cfg.
    for (b = 0; b < l.batch; ++b){
        for(i = 0; i < l.h*l.w*l.n; ++i){
            int index = size*i + b*l.outputs;
            softmax(l.output + index + 5, l.classes, 1, l.output + index + 5);
            }
        }
    for (b = 0; b < l.batch; ++b){
        printf("size = %d outputs = %d\n",size,l.outputs);
        for(i = 0; i < l.h*l.w*l.n; ++i){
            int index = size*i + b*l.outputs;
            l.output[index + 4] = logistic_activate(l.output[index + 4]);
            0 && printf("output %d = %5.4f\n",index+4,l.output[index+4]);
            }
        }
    if (SHOW_INPUT) {
        printf("region output\n");
        for (int i = 0; i < l.outputs; i++) {
            printf("[%4d] = %5.4f\n",i,l.output[i]);
            }
        }
    }

/*
   obj\darknet.exe detect cfg\yolo.cfg v2/yolo.weights data/dog.jpg

yolo.cfg says "[region]"

darknet.c:
    calls test_detector 
detector.c
    runs the network (calls network_predict)
    calls get_region_boxes in region_layer.c
 */

void test_yolo(float *fresult,  int num_results, int h, int w, int img_h, int img_w) {
    Layer L = make_region_layer(1,w,h,
        Region_parms::num,  Region_parms::classes, Region_parms::coords);
    forward_region_layer(L,fresult);
    int classes = L.classes;
    int total = L.w*L.h*L.n;
    Box *boxes = new Box[total];
    // There are total arrays each of which has classes probabilities.
    // Allocate them all together.
    int num_probs = total*(classes+1);	// detector.c
    Prob **probs = new Prob*[total];
    Prob *prob_all = new Prob[num_probs];
    memset(prob_all,0,num_probs*sizeof(Prob));
    {
    Prob *p = prob_all;
    for (int j = 0; j < total; j++) {
        probs[j] = p;
        p += classes + 1;
        // Done in memset above.  Not sure the memset is at all needed.
        // for (int c = 0; c < classes; c++) probs[j][c] = 0;
        }
    }
    float hier_thresh = 0.5;	// detector.c
    float thresh = .24;		// detector.c
    get_region_boxes(L, 1, 1, thresh, probs, boxes, 0, 0, hier_thresh);
    float nms = Region_parms::nms;
    if (nms) {
        do_nms_sort(boxes, probs, total, L.classes, nms);
        }
    Image **alphabet = 0;
    // Input image height and width.
    // Darknet would draw labels on top of the input image.
    Image im = {(short)img_h, (short)img_w};
    draw_detections(im, total, thresh, boxes, probs, voc_names, alphabet, 
        sizeof(voc_names)/sizeof(*voc_names));
    delete boxes;
    delete probs;
    delete prob_all;
    }


// data_y, data_x is the original dimension of the image, prior
// to network-size conversion.
#define IN_SIZE_formal int noutputs, int data_ch, int data_y, int data_x
#define IN_SIZE_actual noutputs, data_ch, data_y, data_x

// For yolo, the result dimensions are 1470 x 1 x 1 (1470 "channels").
//
template <typename data_type>
     struct Blob_and_size {
        const char *name;       // blob name
        const char *layer_name; // layer generating this blob
        const char *layer_type; // type of layer
        unsigned size;          // total size in bytes
        unsigned element_size;  // element size in bytes (contains the pixel)
        unsigned pixel_size;    // blob pixel size in bits
        unsigned Z,Y,X;         // dimensions.
        data_type *blob;        // ptr to blob
        double scale;		// scale of blob
	bool is_signed;
	short int zero_point;
        int num_pixels() { return size/element_size; }
        };


template <typename T>
void yolo_fixed_T(void *outputs,  IN_SIZE_formal) {
    Blob_and_size<T> &output = *((Blob_and_size<T> **)outputs)[0];
    int total = output.Z*output.Y*output.X;
    Result *fresult = new Result[total];
    for (int i = 0; i < total; i++) {
        fresult[i] = float(output.blob[i]-output.zero_point)/output.scale;
        0 && printf("fr[%d]=%f\n",i,fresult[i]);
        }
    test_yolo(fresult,output.Z,output.Y,output.X,data_y,data_x);
    delete fresult;
    }
void yolo_fixed(void *outputs, IN_SIZE_formal) {
    // temp is just to get the size and type of the blob.
    Blob_and_size<short> &temp = *((Blob_and_size<short> **)outputs)[0];
    0 && printf("temp size:%d signed:%d\n",temp.pixel_size,temp.is_signed);
    if (temp.pixel_size == 8)
        if (temp.is_signed) yolo_fixed_T<signed char>(outputs,IN_SIZE_actual);
	else yolo_fixed_T<unsigned char>(outputs,IN_SIZE_actual);
    else
        // Only other possibility is short.
	yolo_fixed_T<short>(outputs,IN_SIZE_actual);
    }

extern void yolo_float(void *outputs, IN_SIZE_formal) {
    Blob_and_size<float> &output = *((Blob_and_size<float> **)outputs)[0];
    test_yolo(output.blob,output.Z,output.Y,output.X,data_y,data_x);
    }

extern void yolo_double(void *outputs, IN_SIZE_formal) {
    Blob_and_size<double> &output = *((Blob_and_size<double> **)outputs)[0];
    int total = output.Z*output.Y*output.X;
    Result *fresult = new Result[total];
    for (int i = 0; i < total; i++) {
        fresult[i] = output.blob[i];	// From double to float.
        }
    test_yolo(fresult,output.Z,output.Y,output.X,data_y,data_x);
    delete fresult;
    }

#if TEST
#include "input.c"
void main() {
    test_yolo(fresult,sizeof(fresult)/sizeof(*fresult));
    }
#endif

