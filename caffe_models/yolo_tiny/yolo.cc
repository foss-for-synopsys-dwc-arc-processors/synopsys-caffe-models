#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Detection layer parmameers.
namespace Detection_parms {
    static const int num=2;
    static const int side=7;
    static const int classes=20;
    static const int sqrt=1;
    // Not specified in the prototxt, but hard-coded.  See src/yolo.c from darknet.
    static const float thresh = 0.2;
    static const float nms = 0.4;
    }
struct Box {
    float x, y, w, h;
    };
typedef float Prob;
typedef float Result;

struct Layer {
    int side;
    int n;
    int sqrt;
    Result *output;
    int classes;
    };

void get_detection_boxes(
        Layer &l, int w, int h, 
        float thresh, 
        Prob **probs, Box *boxes, int only_objectness) {
    // output = 1470 = 30 x 7 x 7.
    Result *predictions = l.output;
    //int per_cell = 5*num+classes;
    for (int i = 0; i < l.side*l.side; i++) {
        int row = i / l.side;
        int col = i % l.side;
        for (int n = 0; n < l.n; ++n) {
            int index = i*l.n + n;
            int p_index = l.side*l.side*l.classes + i*l.n + n;
            Result scale = predictions[p_index];
            0 && printf("pred[%d]=%f\n", p_index, scale);
            int box_index = l.side*l.side*(l.classes + l.n) + (i*l.n + n)*4;
            boxes[index].x = (predictions[box_index + 0] + col) / l.side * w;
            boxes[index].y = (predictions[box_index + 1] + row) / l.side * h;
            boxes[index].w = pow(predictions[box_index + 2], (l.sqrt?2:1)) * w;
            boxes[index].h = pow(predictions[box_index + 3], (l.sqrt?2:1)) * h;
            for (int j = 0; j < l.classes; j++) {
                int class_index = i*l.classes;
                Prob prob = scale*predictions[class_index+j];
                if (prob <= thresh) prob = 0;
                probs[index][j] = prob;
                if (prob) 
                    printf("probs[%d][%d] = %f\n",index,j,probs[index][j]);
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

static char *voc_names[] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle", 
    "bus", "car", "cat", "chair", "cow", 
    "diningtable", "dog", "horse", "motorbike", "person", 
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"};

int max_index(Prob *a, int n) {
    if(n <= 0) return -1;
    int i, max_i = 0;
    Prob max = a[0];
    for(i = 1; i < n; ++i){
        if(a[i] > max){
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

void test_yolo(float *fresult,  int num_results, int img_y, int img_x) {
    int side = Detection_parms::side;
    int num = Detection_parms::num;
    int classes = Detection_parms::classes;
    int total = side*side*num;
    Box *boxes = new Box[total];
    // There are total arrays each of which has classes probabilities.
    // Allocate them all together.
    Prob *prob_all = new Prob[total*classes];
    memset(prob_all,0,total*classes*sizeof(Prob));
    Prob **probs = new Prob*[total];
    for (int j = 0; j < total; j++) {
        probs[j] = prob_all + j*classes;
        // Done in memset above.  Not sure the memset is at all needed.
        // for (int c = 0; c < classes; c++) probs[j][c] = 0;
        }
    Layer L;
    L.side = side;
    L.output = fresult;
    L.n = num;
    L.classes = classes;
    L.sqrt = Detection_parms::sqrt;
    float thresh = Detection_parms::thresh;
    get_detection_boxes(L, 1, 1, thresh, probs, boxes, 0);
    float nms = Detection_parms::nms;
    if (nms) {
        do_nms_sort(boxes, probs, L.side*L.side*L.n, L.classes, nms);
        }
    Image **alphabet = 0;
    // Input image height and width.
    // Darknet would draw labels on top of the input image.
    Image im = {(short)img_y,(short)img_x};
    draw_detections(im, L.side*L.side*L.n, thresh, boxes, probs, voc_names, alphabet, 20);
    delete boxes;
    delete probs;
    delete prob_all;
    }


#define IN_SIZE int noutputs, int data_ch, int data_y, int data_x

// For yolo, the result dimensions are 1470 x 1 x 1 (1470 "channels").

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

// This uses --post_verify_func v2:yolo so we can get the zero point.

extern void yolo_float(void *_outputs,  IN_SIZE) {
    typedef Blob_and_size<float> BS;
    BS *output = ((BS **)_outputs)[0];
    test_yolo(output->blob,noutputs,data_y,data_x);
    }

template <typename T> void yolo_convert(void *_outputs, IN_SIZE) {
    typedef Blob_and_size<T> BS;
    BS *output = ((BS **)_outputs)[0];
    int pixels = output->num_pixels();
    Result *fresult = new Result[pixels];
    int ZP = output->zero_point;
    printf("scale is %f zp %d\n",output->scale,ZP);
    for (int i = 0; i < pixels; i++) fresult[i] = (output->blob[i]-ZP)/output->scale;
    test_yolo(fresult,noutputs,data_y,data_x);
    }

extern void yolo_double(void *_outputs,  IN_SIZE) {
    yolo_convert<double>(_outputs,noutputs,data_ch,data_y,data_x);
    }

extern void yolo_fixed(void *_outputs, IN_SIZE) {
    yolo_convert<short>(_outputs,noutputs,data_ch,data_y,data_x);
    }

#if TEST
#include "input.c"
void main() {
    test_yolo(fresult,sizeof(fresult)/sizeof(*fresult));
    }
#endif

