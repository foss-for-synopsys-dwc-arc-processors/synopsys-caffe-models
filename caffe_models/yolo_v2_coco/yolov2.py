
import caffe
import darknet

class darknet_reorg(caffe.Layer):
    def setup(self,bottom,top):
        print "python setup being called"
	self.stride=2

    def reshape(self,bottom,top):
        S = bottom[0].data.shape
        print "shape of bottom is ",S
	print "type of shape is ",type(S)
	# top should be (1, 2048, 13, 13)
        print "python reshape being called"
	# You have to reshape with a * in front of the tuple to split its elements.
	# See https://github.com/NVIDIA/DIGITS/tree/master/examples/python-layer
	# Why I don't know.
	# top[0].reshape(*(S[0],S[1]*4,S[2]/2,S[3]/2))
	# reshape takes 4 integers.
	s = self.stride
	top[0].reshape(S[0],S[1]*s*s,S[2]/s,S[3]/s)

    def forward(self,bottom,top):
        print "python forward being called"
        print "bottom shape is ",bottom[0].data.shape
	print "tb is ",type(bottom[0].data)
        print "top    shape is ",top[0].data.shape
	BS = bottom[0].data.shape
	darknet.reorg(bottom[0].data,BS[3],BS[2],BS[1],BS[0],self.stride,top[0].data)
	# print "input was ",bottom[0].data
	# print "output is ",top[0].data

    def backward(self,bottom,top):
        raise Exception("python backward being called")

    @staticmethod
    def layer_implementation_text(blob_parameters,layer_fnum):
        B = blob_parameters[1]	# Take shape of input.
	sh = B.shape
	STRIDE = 2
	S = ""
	S += "    typedef data_type *DS;\n"
        S += "    darknet_reorg<{},{},{},1,{}>".format(sh[3],sh[2],sh[1],STRIDE)
	S += "(DS("+blob_parameters[1].cname+"),DS("+blob_parameters[0].cname+"));"
	return S

