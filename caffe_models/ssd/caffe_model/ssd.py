
import caffe
print "in pythonmodule.py"

class subgraph_g1(caffe.Layer):
    def setup(self,bottom,top):
        print "python setup being called"

    def reshape(self,bottom,top):
        print "python reshape being called"
        # top[0] reshape is a hack to get python layer working, it doesn't produce correct result
        top[0].reshape(*bottom[0].data.shape)

    def forward(self,bottom,top):
        print "python forward being called"

    def backward(self,bottom,top):
        raise Exception("python backward being called")

    @staticmethod
    def layer_implementation_text(blob_parameters,layer_fnum):
        return ""
