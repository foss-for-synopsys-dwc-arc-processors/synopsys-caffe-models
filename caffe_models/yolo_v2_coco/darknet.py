import numpy as np


def reorg(input, w, h, c, batch, stride, output):
    # This implements darknet's bizarre reorg layer going forward.
    # e.g. input is 512 x 26 x 26  output is 2048 x 13 x 13
    # darknet treats input as 128 x 52 x 52 and output as if it were the 
    # input dimensions.  This is probably a bug in darknet.
    # c x h x w is the input.
    # Operate on a weird reshape.
    out_c = c/(stride*stride);
    print "out_c is ",out_c
    # Can reshape just by clobbering the shape.
    inshape = input.shape
    print "inshape is ",inshape
    print "outshape is ",output.shape
    input.shape=(batch,out_c,h*stride,w*stride)
    output.shape = inshape
    for b in range(0,batch):
	for k in range(0,c):
	    for y in range(0,h):
		for x in range(0,w):
		    in_index  = x + w*(y + h*(k + c*b));
		    c2 = k % out_c;
		    offset = k / out_c;
		    w2 = x*stride + offset % stride;
		    h2 = y*stride + offset / stride;
		    in_pixel = input[b][c2][h2][w2];
		    output[b][k][y][x] = in_pixel;
    ns = (batch,c*stride*stride,h/stride,w/stride)
    # print "trying for new shape ",ns
    output.shape = ns

def test():
    BATCH=1
    CHAN=4
    Y=6
    X=6
    STRIDE=2

    def func():
	ar = np.zeros(shape=(BATCH*CHAN*Y*X))
	num = .0
	for x in range(0,BATCH*CHAN*Y*X):
	    ar[x] = num
	    num += .01
	print ar
	ar.shape=(BATCH,CHAN,Y,X)
	print ar
	return ar
	    
    ar = func()
    ab = np.zeros(shape=(BATCH,CHAN*STRIDE*STRIDE,Y/STRIDE,Y/STRIDE))
    darknet_reorg(ar,X,Y,CHAN,BATCH,STRIDE,ab)
    print ab

# test()
