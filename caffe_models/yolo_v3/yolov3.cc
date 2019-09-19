#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

// This code compiles for tiny yolo
// unless you -DFULL_YOLO.

struct Box {
    float x, y, w, h;
    };
void pbox(Box *b) {
    printf("!box: %f %f %f %f\n", b->x, b->y, b->w, b->h);
    }

typedef float Prob;
typedef float Result;

#define PRINTLINE printf("LINE %d\n",__LINE__);

static const char YOLO = 'y';	// layer type.

static const bool trace = false;

struct Layer {
    int batch;
    int side;
    int n;		// number of output maps in a layer.
    int outputs; 	// Total number of output pixels: maps x h x w
    int inputs;
    int sqrt;
    Result *output;
    int classes;
    int c, h, w;	// Input to a layer.
    int out_c, out_h, out_w;
    Result *delta;
    Result *cost, *biases;
    // Needed for the yolo layer.
    char type;
    int total;	// yolo num value
    int *mask;
    float *bias_updates;
    int truths;
    int coords;
    };
    
struct Detection{
    Box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
    };

struct Network {
    int h, w;	// network dimensions.
    float *input;
    Layer *layers;
    int n;	// # of layers.
    };

#if 0
void resize_yolo_layer(Layer *l, int w, int h) {
    l->w = w;
    l->h = h;

    l->outputs = h*w*l->n*(l->classes + 4 + 1);
    l->inputs = l->outputs;

    l->output = realloc(l->output, l->batch*l->outputs*sizeof(float));
    l->delta = realloc(l->delta, l->batch*l->outputs*sizeof(float));

    }
#endif

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

Box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride) {
0 && printf("!get yolo box n %d index %d i %d j %d lw %d lh %d w %d h %d stride %d\n",
 n, index, i, j, lw, lh, w, h, stride);
    Box b;
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
    }

float delta_yolo_box(Box truth, float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, float *delta, float scale, int stride) {
    Box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);

    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);

    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
    }


void delta_yolo_class(float *output, float *delta, int index, int klass, int classes, int stride, float *avg_cat) {
    int n;
    if (delta[index]) {
        delta[index + stride*klass] = 1 - output[index + stride*klass];
        if (avg_cat) *avg_cat += output[index + stride*klass];
        return;
        }
    for (n = 0; n < classes; ++n) {
        delta[index + stride*n] = ((n == klass)?1 : 0) - output[index + stride*n];
        if (n == klass && avg_cat) *avg_cat += output[index + stride*n];
        }
    }

void correct_yolo_boxes(Detection *dets, int n, int w, int h, int netw, int neth, int relative) {
    int i;
    int new_w=0;
    int new_h=0;
    if (((float)netw/w) < ((float)neth/h)) {
        new_w = netw;
        new_h = (h * netw)/w;
        } 
    else {
        new_h = neth;
        new_w = (w * neth)/h;
        }
    for (i = 0; i < n; ++i) {
        Box b = dets[i].bbox;
        b.x =  (b.x - (netw - new_w)/2./netw) / ((float)new_w/netw); 
        b.y =  (b.y - (neth - new_h)/2./neth) / ((float)new_h/neth); 
        b.w *= (float)netw/new_w;
        b.h *= (float)neth/new_h;
        if (!relative) {
            b.x *= w;
            b.w *= w;
            b.y *= h;
            b.h *= h;
            }
        dets[i].bbox = b;
        }
    }

void avg_flipped_yolo(Layer l) {
    int i,j,n,z;
    float *flip = l.output + l.outputs;
    for (j = 0; j < l.h; ++j) {
        for (i = 0; i < l.w/2; ++i) {
            for (n = 0; n < l.n; ++n) {
                for (z = 0; z < l.classes + 4 + 1; ++z) {
                    int i1 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + i;
                    int i2 = z*l.w*l.h*l.n + n*l.w*l.h + j*l.w + (l.w - i - 1);
                    float swap = flip[i1];
                    flip[i1] = flip[i2];
                    flip[i2] = swap;
                    if (z == 0) {
                        flip[i1] = -flip[i1];
                        flip[i2] = -flip[i2];
                        }
                    }
                }
            }
        }
    for (i = 0; i < l.outputs; ++i) {
        l.output[i] = (l.output[i] + flip[i])/2.;
        }
    }

static int entry_index(Layer l, int batch, int location, int entry) {
    int n =   location / (l.w*l.h);
    int loc = location % (l.w*l.h);
    return batch*l.outputs + n*l.w*l.h*(4+l.classes+1) + entry*l.w*l.h + loc;
    }

int get_yolo_detections(
        Layer l, int w, int h, int netw, int neth, 
        float thresh, int *map, int relative, Detection *dets) {
    int i,j,n;
    float *predictions = l.output;
    if (l.batch == 2) avg_flipped_yolo(l);
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i) {
        int row = i / l.w;
        int col = i % l.w;
        for (n = 0; n < l.n; ++n) {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            float objectness = predictions[obj_index];
            if (objectness <= thresh) continue;
            int box_index  = entry_index(l, 0, n*l.w*l.h + i, 0);
            dets[count].bbox = get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w*l.h);
            dets[count].objectness = objectness;
            dets[count].classes = l.classes;
            for (j = 0; j < l.classes; ++j) {
                int class_index = entry_index(l, 0, n*l.w*l.h + i, 4 + 1 + j);
                float prob = objectness*predictions[class_index];
                dets[count].prob[j] = (prob > thresh) ? prob : 0;
                }
            ++count;
            }
        }
    correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
    return count;
    }

enum ACTIVATION {
    LOGISTIC, RELU, RELIE, LINEAR, RAMP, TANH, PLSE, 
    LEAKY, ELU, LOGGY, STAIR, HARDTAN, LHTAN, SELU
    };

static inline float logistic_activate(float x){return 1./(1. + exp(-x));}

float activate(float x, ACTIVATION a) {
    switch(a){
        case LOGISTIC: return logistic_activate(x);
        default:
            printf("activation not supported\n");
            exit(1);
        }
    }

void activate_array(float *x, const int n, const ACTIVATION a) {
    int i;
    for (i = 0; i < n; ++i){
        x[i] = activate(x[i], a);
        }
    }

void forward_yolo_layer(const Layer l, Network net) {
    int i,j,b,t,n;
    // Copy input to output.  net.input is the input from the layer feeding
    // the yolo layer.
    trace && printf("!fwd yolo layer.  outputs=%d n=%d\n",l.outputs,l.n);
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));
    const int cnt = trace ? 10 : 0;
    for (int i = 0; i < cnt; i++) {
        printf("!input[%d] = %8.4f\n",i,net.input[i]);
        }

    b = 0;	// batch is always 1
    for (n = 0; n < l.n; ++n) {
        int index = entry_index(l, b, n*l.w*l.h, 0);
        trace && printf("!n=%d ix=%d\n",n,index);
        activate_array(l.output + index, 2*l.w*l.h, LOGISTIC);
        index = entry_index(l, b, n*l.w*l.h, 4);
        activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC);
        }
    for (int i = 0; i < cnt; i++) {
        printf("!output[%d] = %f\n",i,l.output[i]);
        }

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    }

void fill_network_boxes(
        Network *net, int w, int h, float thresh, float hier, 
        int *map, int relative, Detection *dets) {
    int j;
    for (j = 0; j < net->n; ++j){
        Layer l = net->layers[j];
        if (l.type == YOLO){
            trace && printf("filling nboxes for yolo j=%d netw %d h %d\n",j,
                    net->w,net->h);
            int count = get_yolo_detections(l, w, h, net->w, net->h, thresh, map, relative, dets);
            dets += count;
            }
        }
    }

int yolo_num_detections(Layer l, float thresh) {
    int i, n;
    int count = 0;
    for (i = 0; i < l.w*l.h; ++i) {
        for (n = 0; n < l.n; ++n) {
            int obj_index  = entry_index(l, 0, n*l.w*l.h + i, 4);
            if (l.output[obj_index] > thresh) {
                trace && printf("!o %f > %f\n",l.output[obj_index] , thresh);
                ++count;
                }
            }
        }
    trace && printf("!I have %d detections\n",count);
    return count;
    }
int num_detections(Network *net, float thresh) {
    int i;
    int s = 0;
    for (i = 0; i < net->n; ++i){
        Layer l = net->layers[i];
        if (l.type == YOLO){
            s += yolo_num_detections(l, thresh);
            }
        }
    return s;
    }

inline float *falloc(int a) { 
    0 && printf("!falloc %d by %d\n",a,sizeof(float));
    return (float*)calloc(a,sizeof(float)); 
    }

Detection *make_network_boxes(Network *net, float thresh, int *num) {
    Layer l = net->layers[net->n - 1];
    int i;
    int nboxes = num_detections(net, thresh);
    if (num) *num = nboxes;
    Detection *dets = (Detection *)calloc(nboxes, sizeof(Detection));
    for (i = 0; i < nboxes; ++i){
        dets[i].prob = falloc(l.classes);
        if (l.coords > 4){
            dets[i].mask = falloc(l.coords-4);
            }
        }
    return dets;
    }

Detection *get_network_boxes(
        Network *net, int w, int h, 
        float thresh, float hier, int *map, int relative, int *num) {
    Detection *dets = make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, dets);
    return dets;
    }

Layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes) {
    0 && printf("!make yolo layer w %d h %d n %d total %d\n",w,h,n,total);
    // [yolo] parm:
    // num => total
    // number of masks => n
    // anchors go into l.biases, I suppose a convenient place to store them.
    // There are 2*num anchor pairs.
    int i;
    Layer l = {0};
    l.type = YOLO;

    l.n = n;
    l.total = total;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1);
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = falloc(1);
    l.biases = falloc(total*2);
    if (mask) l.mask = mask;
    else {
        l.mask = (int*)calloc(n, sizeof(int));
        for (i = 0; i < n; ++i) {
            l.mask[i] = i;
            }
        }
    l.bias_updates = falloc(n*2);
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = falloc(batch*l.outputs);
    l.output = falloc(batch*l.outputs);
    for (i = 0; i < total*2; ++i) {
        l.biases[i] = .5;
        }

    //l.forward = forward_yolo_layer;
    //l.backward = backward_yolo_layer;
    return l;
    }

int nms_comparator(const void *pa, const void *pb) {
    Detection a = *(Detection *)pa;
    Detection b = *(Detection *)pb;
    float diff = 0;
    if (b.sort_class >= 0){
        diff = a.prob[b.sort_class] - b.prob[b.sort_class];
        } 
    else {
        diff = a.objectness - b.objectness;
        }
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
    }

void do_nms_sort(Detection *dets, int total, int classes, float thresh) {
    int i, j, k;
    k = total-1;
    for (i = 0; i <= k; ++i){
        if (dets[i].objectness == 0){
            Detection swap = dets[i];
            dets[i] = dets[k];
            dets[k] = swap;
            --k;
            --i;
            }
        }
    total = k+1;

    for (k = 0; k < classes; ++k){
        for (i = 0; i < total; ++i){
            dets[i].sort_class = k;
            }
        qsort(dets, total, sizeof(Detection), nms_comparator);
        for (i = 0; i < total; ++i){
            if (dets[i].prob[k] == 0) continue;
            Box a = dets[i].bbox;
            for (j = i+1; j < total; ++j){
                Box b = dets[j].bbox;
                if (box_iou(a, b) > thresh){
                    dets[j].prob[k] = 0;
                    }
                }
            }
        }
    }


// Strange that tiny yolo v3 has classes=80 but expects to use voc_names.

static const char *voc_names[] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

static const char *coco_names[] = {
    "person", "bicycle", "car", "motorbike", "aeroplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush", };

static const int num_coco_classes = sizeof(coco_names)/sizeof(*coco_names);

static short int coco_categories[num_coco_classes] = {
    // The coco annotations file uses numbers different from 0..79 to
    // denote the categories,  They range from 1 to 90.
    // We set the translation here.
    // This info was obtained by reading instances_val2017.json and looking
    // at the 'categories' property. 
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
    35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
    56, 57, 58, 59, 60, 61, 62, 63, 64, 65,
    67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
    80, 81, 82, 84, 85, 86, 87, 88, 89, 90,
    };

struct Image {
    int h,w;
    };

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

// Collect results in json for ease of post-processing in javascript or python.
//

struct Results {
    // coco annotations are in flt but darknet's drawing code used truncated ints.
    // We might change this later to support the flt pt box.
    typedef int Coord;
    void record_box(
	const char *name, int klass, float prob, int x1, int y1, int x2, int y2) {
	// coco annotations are x1 y1 w h, not x2 y2.
	Result R = {klass,prob,x1,y1,x2,y2};
	R.name = name;
	R.print();
	if (next >= results.size()) {
	    results.resize(5+2*results.size());
	    }
	results[next++] = R;
	}
    struct Result {
        //--------- These entered positionally.
	int klass; float prob;
        Coord x1,y1,x2,y2;
	//----------
	const char *name;
	void print() {
	    printf("    draw box %d %d %d %d %d %d (= x1 y1 x2 y2 w h)\n", 
		x1, y1, x2, y2, x2-x1, y2-y1);
	    }
	void json(std::string &S) {
	    // The python json parser is buggy; it doesn't allow trailing commas.
	    // So we have to prevent generating them here.
	    char Q = '"';
	    char buf[256];
	    const char *bbox = "\"bbox\"";	// coco bbox: x y and size.
	    S += "{";
	    sprintf(buf,"%s : [ %d, %d, %d, %d ]",bbox,x1,y1,x2-x1,y2-y1);
	    S += buf;

	    const char *dncat = "\"darknet_category\"";
	    sprintf(buf,", %s : %d",dncat,klass);
	    S += buf;

	    const char *ccat = "\"category_id\"";	// It's what coco calls it.
	    sprintf(buf,", %s : %d",ccat,coco_categories[klass]);
	    S += buf;

	    const char *name = "\"name\"";
	    sprintf(buf,", %s : %c%s%c",name,Q,this->name,Q);
	    S += buf;
	    
	    const char *prob = "\"prob\"";
	    sprintf(buf,", %s : %.4f",prob,this->prob);
	    S += buf;
	    
	    S += "}";
	    }
        };
    int next;
    Results() { next = 0; }
    std::vector<Result> results;
    std::string json() {
        std::string S;
	// Array of results
	S += "[\n";
	for (int i = 0; i < next; i++) {
	    results[i].json(S);
	    if (i != next-1) S += ',';
	    S += "\n";
	    }
	S += "]\n";
	return S;
        }
    } results;


void draw_detections(Image im, Detection *dets, int num, 
        float thresh, const char **names, Image **alphabet, int classes) {
    trace && printf("!draw detections for im.h %d im.w %d\n",im.h,im.w);

    for (int i = 0; i < num; ++i){
        char labelstr[4096] = {0};
        int klass = -1, prob;
	// A box can belong to more than one klass.
	auto record_result = [] (Image im, Box &b, int klass, const char *name, float prob) {
            // pbox(&b);
            //printf("%f %f %f %f\n", b.x, b.y, b.w, b.h);

            int left  = (b.x-b.w/2.)*im.w;
            int right = (b.x+b.w/2.)*im.w;
            int top   = (b.y-b.h/2.)*im.h;
            int bot   = (b.y+b.h/2.)*im.h;

            if (left < 0) left = 0;
            if (right > im.w-1) right = im.w-1;
            if (top < 0) top = 0;
            if (bot > im.h-1) bot = im.h-1;

            results.record_box(name, klass, prob, left, top, right, bot);
	    };
        for (int cix = 0; cix < classes; ++cix) {
            if (dets[i].prob[cix] > thresh){
                if (klass < 0) {
                    strcat(labelstr, names[cix]);
                    klass = cix;
                    } 
                else {
                    strcat(labelstr, ", ");
                    strcat(labelstr, names[cix]);
                    }
		// Darknet's code prints all the class names & probabilities
		// ahead of a single box.
                printf("%s: %.2f%%\n", names[cix], dets[i].prob[cix]*100);
		// Different from darknet, the box is drawn once for
		// each category it is in.
		record_result(im, dets[i].bbox, cix, names[cix], dets[i].prob[cix]);
                }
            }
        }
    }

void free_detections(Detection *dets, int n) {
    int i;
    for (i = 0; i < n; ++i){
        free(dets[i].prob);
        if (dets[i].mask) free(dets[i].mask);
        }
    free(dets);
    }

namespace Yolo_parms {
    // Not specified in the prototxt, but hard-coded.  See src/detector.c from darknet.
    // (darknet version as of 18Nov04).
    static const float nms = 0.45;	// examples/detector.c darknet.
    static const int classes = num_coco_classes;
#if FULL_YOLO
    static const int num = 9;	// num.  Terrible name for a constant.
    // This matches the network size in yolov3.cfg.
    static const int network_dimension = 608;
#else
    static const int num = 6;
    // This matches the network size in yolov3-tiny.cfg.
    static const int network_dimension = 416;
#endif
    }


void test_yolo(Network *net, Image &im) {
#if FULL_YOLO
    printf("Testing FULL yolo.\n");
#else
    printf("Testing TINY yolo.\n");
#endif
    static const float thresh = 0.5;	// darknet command line -thresh default 0.5.
    const float hier_thresh = 0.5;	// From darknet.c: test_detector call.

    int nboxes = 0;
    Detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 
        0, 1, &nboxes);
    float nms = Yolo_parms::nms;
    Layer l = net->layers[net->n-1];
    // Why are there two yolo layers but we do nms etc only on the last one?
    if (nms) 
        if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    Image **alphabet = 0;
    draw_detections(im, dets, nboxes, thresh, coco_names, alphabet, l.classes);
    free_detections(dets, nboxes);
    #if JSON
    // json result is used for post-processing to compute precision.
    printf("<JSON>\n%s</JSON\>\n",results.json().c_str());
    #endif
    }

#define IN_SIZE int noutputs, int data_ch, int data_y, int data_x

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
        int num_pixels() { return size/element_size; }
        };

void convert_anchors_to_biases(Layer &L,const int *anchors,int nanchors) {
    for (int i = 0; i < nanchors; i++) L.biases[i] = anchors[i];
    }

template <typename data_type>
void run_yolo_layer_with_input(
        Network &net, int lnum, Blob_and_size<data_type>&B, bool convert) {
    if (convert) {
        for (int i = 0; i < B.num_pixels(); i++) 
            net.input[i] = B.blob[i]/B.scale;
        }
    else 
        net.input = (float*)B.blob; 
    forward_yolo_layer(net.layers[lnum],net);
    }

template <typename data_type>
Layer make_yolo_layer(Blob_and_size<data_type> &B, int mask[]) {
    Layer L = make_yolo_layer(
        1, B.X,B.Y,3,	// 3 = number of mask entries
        Yolo_parms::num, mask, Yolo_parms::classes);
    L.output = falloc(L.outputs);
    return L;
    }

template <typename data_type>
void run_tiny_yolo(
        Blob_and_size<data_type> **outputs, IN_SIZE, bool fixed = false) {
    typedef Blob_and_size<data_type> BS;
    // Create a yolo layer for the two outputs.
    // From examples/detector.c
    Network net; 
    // Network dimensions.  This is fixed in the graph.
    net.h = net.w = Yolo_parms::network_dimension;
    // Image dimensions.
    Image im = {data_y,data_x};

    static const int anchors[] = 
#if FULL_YOLO
        {10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326};
#else	// tiny
        {10,14,  23,27,  37,58,  81,82,  135,169,  344,319};
#endif

#if !FULL_YOLO
    // Create the two yolo layers.
    
    // We happen to know that the output list from the tool is backwards; last 
    // yolo layer appers first in the list.
    BS &B0 = *outputs[1];
    BS &B1 = *outputs[0];
    BS *ptrs[]= {&B0,&B1};

    /*
        [yolo]
        mask = 3,4,5
        anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
        classes=80 num=6 

        [yolo]
        mask = 0,1,2
        anchors = 10,14,  23,27,  37,58,  81,82,  135,169,  344,319
        classes=80 num=6 
    */
    int M0[] = {3,4,5}, M1[] = {0,1,2}; 
    static Layer layers[] = {
        make_yolo_layer<data_type>(B0,M0),
        make_yolo_layer<data_type>(B1,M1) };

#else
    // Entries in this array are in a weird order.
    BS &B0 = *outputs[0];	// layer82
    BS &B1 = *outputs[2];	// layer94
    BS &B2 = *outputs[1];	// layer106
    BS *ptrs[] = {&B0,&B1,&B2};

    int M0[] = {6,7,8}, M1[] = {3,4,5}, M2[] = {0,1,2}; 
    static Layer layers[] = {
        make_yolo_layer<data_type>(B0,M0),
        make_yolo_layer<data_type>(B1,M1),
        make_yolo_layer<data_type>(B2,M2)};

#endif
    net.layers = layers;
    net.n = sizeof(layers)/sizeof(*layers);

    bool convert_to_float = sizeof(*B0.blob) != sizeof(float) | fixed;
    if (convert_to_float) {
        // Convert double or int to float.
        int max = 0;
        for (int i = 0; i < noutputs; i++) { 
            int s = outputs[i]->num_pixels();
            if (s > max) max = s;
            }
        0 && printf("!malloc %d floats\n",max);
        net.input = new float[max];
        }

    for (int i = 0; i < net.n; i++) {
        convert_anchors_to_biases(net.layers[i],anchors,sizeof(anchors)/sizeof(*anchors));
        run_yolo_layer_with_input(net,i,*ptrs[i],convert_to_float);
        }

    test_yolo(&net,im);
    if (convert_to_float) delete [] net.input;
    }

extern void yolo_float(void *_outputs,  IN_SIZE) {
    Blob_and_size<float>**outputs = (Blob_and_size<float>**)_outputs;
    run_tiny_yolo<float>(outputs,noutputs,data_ch,data_y,data_x);
    }

extern void yolo_double(void*_outputs, IN_SIZE) {
    Blob_and_size<double>**outputs = (Blob_and_size<double>**)_outputs;
    run_tiny_yolo<double>(outputs,noutputs,data_ch,data_y,data_x);
    }

extern void yolo_fixed(void *_outputs, IN_SIZE) {
    Blob_and_size<short> **outputs = (Blob_and_size<short> **)_outputs;
    0 && printf("ch %d y %d x %d\n",data_ch,data_y,data_x);
    run_tiny_yolo<short>(outputs,noutputs,data_ch,data_y,data_x,true);
    }

#if TEST
int main() {
    Blob_and_size<float> BS[] = {
        { "b1","l1","yolo",0,0,0,255,26,26,(float*)0,1 },
        { "b2","l2","yolo",0,0,0,255,13,13,(float*)0,1 },
        };
    yolo_float(BS,2,3,416,416);
    }
#endif
