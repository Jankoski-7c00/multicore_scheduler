from frontend.onnx_parser import OnnxParser
from classes.tensor import Tensor
from classes.computationNode import ComputationNode
from networkx import DiGraph

class ComputationGraph :
    '''computation graph generate'''
    def __init__(
            self,
            parser: OnnxParser,
            split_factor: int = 4
        ) -> None:
        self.parser = parser
        self.split_factor = split_factor
        self.CN_num = 0
        self.CN_graph = DiGraph()
        self.split_layers = self.split_all_layers()
        self.split_layers_dict = {layer['name']: layer['CN'] for layer in self.split_layers}
        self.generate_CN_DIG()


    def split_relu(self, layer, split_axis: str):
        if layer['op_type'] != 'Relu':
            raise ValueError(f"Wrong op type:{layer['op_type']}, expect: Relu")
        
        N, C, H, W = layer['input_shapes'][0]
        pre_layer = self.parser.find_pre_layer(layer['inputs'][0])
        producer_layer = pre_layer['name']
        consumer_layer = layer['name']
        loop_dim = ('N', 'C', 'H', 'W')
        op = 'Relu'
        CNs = []

        if split_axis == 'channel':
            if C%self.split_factor != 0 :
                raise ValueError('wrong split factor')
            a = C // self.split_factor
            
            for i in range(self.split_factor):
                split_c = i * a
                loop_range = ((0, N-1), (split_c, split_c+a-1), (0, H-1), (0, W-1))
                size = N*H*W*a
                tensor = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer)
                tensor_out = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=consumer_layer)
                CN = ComputationNode(consumer_layer, op)
                CN.tensor_fm.append(tensor)
                CN.tensor_out.append(tensor_out)
                CN.node_ID = self.CN_num
                self.CN_num = self.CN_num + 1
                CN.core_allocation = i
                CNs.append(CN)
        
        elif split_axis == 'height':
            if H%self.split_factor != 0 :
                raise ValueError('wrong split factor')
            a = H // self.split_factor
            
            for i in range(self.split_factor):
                split_h = i * a
                loop_range = ((0, N-1), (0, C-1), (split_h, split_h+a-1), (0, W-1))
                size = N*C*W*a
                tensor = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer)
                tensor_out = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=consumer_layer)
                CN = ComputationNode(consumer_layer, op)
                CN.tensor_fm.append(tensor)
                CN.tensor_out.append(tensor_out)
                CN.node_ID = self.CN_num
                self.CN_num = self.CN_num + 1
                CN.core_allocation = i
                CNs.append(CN)

        else:
            raise ValueError(f'wrong split_axis: {split_axis}')

        split_layer = {}
        split_layer['name'] = consumer_layer
        split_layer['CN'] = CNs
        return split_layer
    
    def split_batchnorm(self, layer, split_axis: str):
        if layer['op_type'] != 'BatchNormalization':
            raise ValueError(f"Wrong op type:{layer['op_type']}, expect: BatchNormalization")
        
        N, C, H, W = layer['input_shapes'][0]
        pre_layer = self.parser.find_pre_layer(layer['inputs'][0])
        producer_layer = pre_layer['name']
        consumer_layer = layer['name']
        loop_dim = ('N', 'C', 'H', 'W')
        w_dim = ('scale', 'B', 'mean', 'var')
        op = 'BatchNormalization'
        CNs = []

        if split_axis == 'channel':
            if C%self.split_factor != 0 :
                raise ValueError('wrong split factor')
            a = C // self.split_factor
            for i in range(self.split_factor):
                split_c = i * a
                loop_range = ((0, N-1), (split_c, split_c+a-1), (0, H-1), (0, W-1))
                loop_range_w = ((split_c, split_c+a-1), (split_c, split_c+a-1), (split_c, split_c+a-1), (split_c, split_c+a-1))
                size = N*H*W*a
                size_w = a*4
                tensor_in = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer)
                tensor_out = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=consumer_layer)
                tensor_w = Tensor(size_w, loop_dimensions=w_dim, loop_ranges=loop_range_w, is_weight=True)
                CN = ComputationNode(consumer_layer, op)
                CN.tensor_fm.append(tensor_in)
                CN.tensor_w.append(tensor_w)
                CN.tensor_out.append(tensor_out)
                CN.node_ID = self.CN_num
                self.CN_num = self.CN_num + 1
                CN.core_allocation = i
                CNs.append(CN)

        elif split_axis == 'height':
            if H%self.split_factor != 0 :
                raise ValueError('wrong split factor')
            a = H // self.split_factor
            size_w = C*4
            loop_range_w = ((0, C-1), (0, C-1), (0, C-1), (0, C-1))
            for i in range(self.split_factor):
                split_h = i * a
                loop_range = ((0, N-1), (0, C-1), (split_h, split_h+a-1), (0, W-1))
                size = N*C*W*a
                tensor_in = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer)
                tensor_out = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=consumer_layer)
                tensor_w = Tensor(size_w, loop_dimensions=w_dim, loop_ranges=loop_range_w, is_weight=True)
                CN = ComputationNode(consumer_layer, op)
                CN.tensor_fm.append(tensor_in)
                CN.tensor_w.append(tensor_w)
                CN.tensor_out.append(tensor_out)
                CN.node_ID = self.CN_num
                self.CN_num = self.CN_num + 1
                CN.core_allocation = i
                CNs.append(CN)
        
        else:
            raise ValueError('wrong split_axis.')
        
        split_layer = {}
        split_layer['name'] = consumer_layer
        split_layer['CN'] = CNs
        return split_layer
        
    def split_add(self, layer, split_axis: str):
        if layer['op_type'] != 'Add':
            raise ValueError(f"Wrong op type:{layer['op_type']}, expect: Add")
        
        N, C, H, W = layer['input_shapes'][0]
        pre_layer_1 = self.parser.find_pre_layer(layer['inputs'][0])
        pre_layer_2 = self.parser.find_pre_layer(layer['inputs'][1])
        producer_layer_1 = pre_layer_1['name']
        producer_layer_2 = pre_layer_2['name']
        consumer_layer = layer['name']
        loop_dim = ('N', 'C', 'H', 'W')
        op = 'Add'
        CNs = []

        if split_axis == 'channel':
            if C%self.split_factor != 0 :
                raise ValueError('wrong split factor')
            a = C // self.split_factor
            
            for i in range(self.split_factor):
                split_c = i * a
                loop_range = ((0, N-1), (split_c, split_c+a-1), (0, H-1), (0, W-1))
                size = N*H*W*a
                tensor_1 = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer_1)
                tensor_2 = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer_2)
                tensor_out = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=consumer_layer)
                CN = ComputationNode(consumer_layer, op)
                CN.tensor_fm.append(tensor_1)
                CN.tensor_fm.append(tensor_2)
                CN.tensor_out.append(tensor_out)
                CN.node_ID = self.CN_num
                self.CN_num = self.CN_num + 1
                CN.core_allocation = i
                CNs.append(CN)
        
        elif split_axis == 'height':
            if H%self.split_factor != 0 :
                raise ValueError('wrong split factor')
            a = H // self.split_factor
            
            for i in range(self.split_factor):
                split_h = i * a
                loop_range = ((0, N-1), (0, C-1), (split_h, split_h+a-1), (0, W-1))
                size = N*C*W*a
                tensor_1 = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer_1)
                tensor_2 = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer_2)
                tensor_out = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=consumer_layer)
                CN = ComputationNode(consumer_layer, op)
                CN.tensor_fm.append(tensor_1)
                CN.tensor_fm.append(tensor_2)
                CN.tensor_out.append(tensor_out)
                CN.node_ID = self.CN_num
                self.CN_num = self.CN_num + 1
                CN.core_allocation = i
                CNs.append(CN)

        else:
            raise ValueError(f'wrong split_axis:{split_axis}.')

        split_layer = {}
        split_layer['name'] = consumer_layer
        split_layer['CN'] = CNs
        return split_layer
        
    def split_conv(self, layer, split_axis: str):
        if layer['op_type'] != 'Conv':
            raise ValueError(f"Wrong op type:{layer['op_type']}, expect: Conv")
        
        N, C, H, W = layer['input_shapes'][0]
        N_out, C_out, H_out, W_out = layer['output_shapes'][0]
        pad_h_top, _, pad_h_bottom, _ = layer['conv_attributes']['pads']
        pre_layer = self.parser.find_pre_layer(layer['inputs'][0])
        if pre_layer == None:
            producer_layer = None
        else:
            producer_layer = pre_layer['name']
        consumer_layer = layer['name']
        loop_dim = ('N', 'C', 'H', 'W')
        loop_dim_w = ('O', 'I', 'H', 'W')
        O, I, H_k, W_k = layer['input_shapes'][1]
        assert O == C_out, 'kernel num != output channel num'
        op = 'Conv'
        CNs = []

        if split_axis == 'channel':
            if O%self.split_factor != 0:
                raise ValueError('wrong split factor')
            a = O // self.split_factor
            
            for i in range(self.split_factor):
                split_c = i * a
                
                loop_range = ((0, N-1), (0, C-1), (0, H-1), (0, W-1))
                size = N*H*W*C
                tensor = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer)
                
                loop_range_out = ((0, N_out-1), (split_c, split_c+a-1), (0, H_out-1), (0, W_out-1))
                size_out = N_out*a*H_out*W_out
                tensor_out = Tensor(size_out, loop_dimensions=loop_dim, loop_ranges=loop_range_out, producer_layer=consumer_layer)
                
                loop_range_w = ((split_c, split_c+a-1), (0, I-1), (0, H_k-1), (0, W_k-1))
                loop_size_w = a*I*H_k*W_k
                tensor_w = Tensor(loop_size_w, loop_dimensions=loop_dim_w, loop_ranges=loop_range_w, is_weight=True)
                
                CN = ComputationNode(consumer_layer, op)
                CN.tensor_fm.append(tensor)
                CN.tensor_out.append(tensor_out)
                CN.tensor_w.append(tensor_w)
                CN.node_ID = self.CN_num
                self.CN_num = self.CN_num + 1
                CN.core_allocation = i
                CNs.append(CN)
        
        elif split_axis == 'height':
            if H%self.split_factor != 0 and H_out%self.split_factor != 0:
                raise ValueError('wrong split factor')
            a = H // self.split_factor
            b = H_out // self.split_factor
            
            for i in range(self.split_factor):
                split_h = i * a
                loop_range = ((0, N-1), (0, C-1), (max(split_h-pad_h_top, 0), min(split_h+a+pad_h_bottom-1, H-1)), (0, W-1))
                size = N*C*W * (min(split_h+a+pad_h_bottom-1, H-1) - max(split_h-pad_h_top, 0) + 1)
                tensor = Tensor(size, loop_dimensions=loop_dim, loop_ranges=loop_range, producer_layer=producer_layer)

                split_h_out = i*b
                loop_range_out = ((0, N_out-1), (0, C_out-1), (split_h_out, split_h_out+b-1), (0, W_out-1))
                size_out = N_out*b*C_out*W_out
                tensor_out = Tensor(size_out, loop_dimensions=loop_dim, loop_ranges=loop_range_out, producer_layer=consumer_layer)

                loop_range_w = ((0, O-1), (0, I-1), (0, H_k-1), (0, W_k-1))
                loop_size_w = O*I*H_k*W_k
                tensor_w = Tensor(loop_size_w, loop_dimensions=loop_dim_w, loop_ranges=loop_range_w, is_weight=True)

                CN = ComputationNode(consumer_layer, op)
                CN.node_ID = self.CN_num
                self.CN_num = self.CN_num + 1
                CN.tensor_fm.append(tensor)
                CN.tensor_out.append(tensor_out)
                CN.tensor_w.append(tensor_w)
                CN.core_allocation = i
                
                CNs.append(CN)

        else:
            raise ValueError('wrong split_axis.')

        split_layer = {}
        split_layer['name'] = consumer_layer
        split_layer['CN'] = CNs
        return split_layer

    def split_maxpool(self, layer, split_axis: str):
        if layer['op_type'] != 'MaxPool':
            raise ValueError(f"Wrong op type:{layer['op_type']}, expect: MaxPool")
        
        N, C, H, W = layer['input_shapes'][0]
        size_in = N*C*H*W
        N_out, C_out, H_out, W_out = layer['output_shapes'][0]
        size_out = N_out*C_out*H_out*W_out
        pre_layer = self.parser.find_pre_layer(layer['inputs'][0])
        producer_layer = pre_layer['name']
        consumer_layer = layer['name']
        loop_dim = ('N', 'C', 'H', 'W')
        tensor_in = Tensor(size_in, loop_dimensions=loop_dim, loop_ranges=((0, N-1), (0, C-1), (0, H-1), (0, W-1)), producer_layer=producer_layer)
        tensor_out = Tensor(size_out, loop_dimensions=loop_dim, loop_ranges=((0, N_out-1), (0, C_out-1), (0, H_out-1), (0, W_out-1)), producer_layer=consumer_layer)
        CN =ComputationNode(consumer_layer, op_type='MaxPool')
        CN.tensor_fm.append(tensor_in)
        CN.tensor_out.append(tensor_out)
        CN.node_ID = self.CN_num
        self.CN_num = self.CN_num + 1
        CN.core_allocation = 0
        CNs = [CN]
        split_layer = {}
        split_layer['name'] = consumer_layer
        split_layer['CN'] = CNs
        return split_layer

    def split_global_average_pool(self, layer, split_axis: str):
        if layer['op_type'] != 'GlobalAveragePool':
            raise ValueError(f"Wrong op type:{layer['op_type']}, expect: GlobalAveragePool")
        
        N, C, H, W = layer['input_shapes'][0]
        size_in = N*C*H*W
        N_out, C_out, H_out, W_out = layer['output_shapes'][0]
        size_out = N_out*C_out*H_out*W_out
        pre_layer = self.parser.find_pre_layer(layer['inputs'][0])
        producer_layer = pre_layer['name']
        consumer_layer = layer['name']
        loop_dim = ('N', 'C', 'H', 'W')
        tensor_in = Tensor(size_in, loop_dimensions=loop_dim, loop_ranges=((0, N-1), (0, C-1), (0, H-1), (0, W-1)), producer_layer=producer_layer)
        tensor_out = Tensor(size_out, loop_dimensions=loop_dim, loop_ranges=((0, N_out-1), (0, C_out-1), (0, H_out-1), (0, W_out-1)), producer_layer=consumer_layer)
        CN =ComputationNode(consumer_layer, op_type='GlobalAveragePool')
        CN.tensor_fm.append(tensor_in)
        CN.tensor_out.append(tensor_out)
        CN.node_ID = self.CN_num
        self.CN_num = self.CN_num + 1
        CN.core_allocation = 0
        CNs = [CN]
        split_layer = {}
        split_layer['name'] = consumer_layer
        split_layer['CN'] = CNs
        return split_layer
    
    def split_flatten(self, layer, split_axis: str):
        if layer['op_type'] != 'Flatten':
            raise ValueError(f"Wrong op type:{layer['op_type']}, expect: Flatten")
        
        N, C, H, W = layer['input_shapes'][0]
        size_in = N*C*H*W
        H_out, W_out = layer['output_shapes'][0]
        size_out = H_out*W_out
        pre_layer = self.parser.find_pre_layer(layer['inputs'][0])
        producer_layer = pre_layer['name']
        consumer_layer = layer['name']
        loop_dim = ('N', 'C', 'H', 'W')
        loop_dim_out = ('H', 'W')
        tensor_in = Tensor(size_in, loop_dimensions=loop_dim, loop_ranges=((0, N-1), (0, C-1), (0, H-1), (0, W-1)), producer_layer=producer_layer)
        tensor_out = Tensor(size_out, loop_dimensions=loop_dim_out, loop_ranges=((0, H_out-1), (0, W_out-1)), producer_layer=consumer_layer)
        CN =ComputationNode(consumer_layer, op_type='Flatten')
        CN.tensor_fm.append(tensor_in)
        CN.tensor_out.append(tensor_out)
        CN.node_ID = self.CN_num
        self.CN_num = self.CN_num + 1
        CN.core_allocation = 0
        CNs = [CN]
        split_layer = {}
        split_layer['name'] = consumer_layer
        split_layer['CN'] = CNs
        return split_layer
    
    def split_gemm(self, layer, split_axis: str):
        if layer['op_type'] != 'Gemm':
            raise ValueError(f"Wrong op type:{layer['op_type']}, expect: Gemm")
        
        H, W = layer['input_shapes'][0]
        size_in = H*W
        H_out, W_out = layer['output_shapes'][0]
        size_out = H_out*W_out
        pre_layer = self.parser.find_pre_layer(layer['inputs'][0])
        producer_layer = pre_layer['name']
        consumer_layer = layer['name']
        loop_dim = ('H', 'W')
        tensor_in = Tensor(size_in, loop_dimensions=loop_dim, loop_ranges=((0, H-1), (0, W-1)), producer_layer=producer_layer)
        tensor_out = Tensor(size_out, loop_dimensions=loop_dim, loop_ranges=((0, H_out-1), (0, W_out-1)), producer_layer=consumer_layer)

        H_w, W_w = layer['input_shapes'][1]
        size_w = H_w*W_w
        tensor_w = Tensor(size_w, loop_dimensions=loop_dim, loop_ranges=((0, H_w-1), (0, W_w-1)), is_weight=True)
        
        CN =ComputationNode(consumer_layer, op_type='Gemm')
        CN.tensor_fm.append(tensor_in)
        CN.tensor_out.append(tensor_out)
        CN.tensor_w.append(tensor_w)

        if layer['attributes']['has_bias'] is True :
            bias_size = layer['input_shapes'][2][0]
            tensor_bias = Tensor(bias_size, loop_dimensions=('W'), loop_ranges=((0, bias_size-1)), is_weight=True)
            CN.tensor_w.append(tensor_bias)
        CN.node_ID = self.CN_num
        self.CN_num = self.CN_num + 1
        CN.core_allocation = 0
        CNs = [CN]
        split_layer = {}
        split_layer['name'] = consumer_layer
        split_layer['CN'] = CNs
        return split_layer
    
    def split_all_layers(self):
        split_layers = []
        for layer in self.parser.layers:
            if layer['input_shapes'][0][1] > 128:
                split_axis = 'channel'
            else:
                split_axis = 'height'

            if layer['op_type'] == 'Relu':
                split_layers.append(self.split_relu(layer, split_axis))
            elif layer['op_type'] == 'Add':
                split_layers.append(self.split_add(layer, split_axis))
            elif layer['op_type'] == 'BatchNormalization':
                split_layers.append(self.split_batchnorm(layer, split_axis))
            elif layer['op_type'] == 'Conv':
                split_layers.append(self.split_conv(layer, split_axis))
            elif layer['op_type'] == 'MaxPool':
                split_layers.append(self.split_maxpool(layer, split_axis))
            elif layer['op_type'] == 'GlobalAveragePool':
                split_layers.append(self.split_global_average_pool(layer, split_axis))
            elif layer['op_type'] == 'Flatten':
                split_layers.append(self.split_flatten(layer, split_axis))
            elif layer['op_type'] == 'Gemm':
                split_layers.append(self.split_gemm(layer, split_axis))
            else :
                raise ValueError(f"wrong op type: {layer['op_type']}")
        
        return split_layers
    
    def generate_CN_DIG(self):
        for split_layer in self.split_layers:
            for cn in split_layer['CN']:
                self.CN_graph.add_node(cn)
        
        def is_dependent(producer: ComputationNode, consumer: ComputationNode) -> bool:
            for tensor_a in producer.tensor_out:
                for tensor_b in consumer.tensor_fm:
                    if tensor_a.producer_layer == tensor_b.producer_layer:
                        range_a = tensor_a.loop_ranges
                        range_b = tensor_b.loop_ranges
                        overlap = all(max(a[0], b[0]) <= min(a[1], b[1]) for a, b in zip(range_a, range_b))
                        if overlap:
                            return True
            return False

        for producer, consumer in self.parser.layer_graph.edges():
            producer_CNs = self.split_layers_dict[producer]
            consumer_CNs = self.split_layers_dict[consumer]
            for producer_cn in producer_CNs:
                for consumer_cn in consumer_CNs:
                    if is_dependent(producer_cn, consumer_cn):
                        self.CN_graph.add_edge(producer_cn, consumer_cn)
        
        isolated_nodes = [node for node in self.CN_graph.nodes() if self.CN_graph.in_degree(node) == 0 and self.CN_graph.out_degree(node) == 0]
        if isolated_nodes:
            raise ValueError(f'there are isolated nodes:{isolated_nodes}.')
        else:
            print("layer split completed!")

#model = onnx.load('/Users/xiangyy/Downloads/resnet50-v1-12.onnx')
#parser = OnnxParser(model)
#print(parser.layer_graph.number_of_edges())
#print(parser.layer_graph.number_of_nodes())
#cg = ComputationGraph(parser)
#print(cg.CN_graph.number_of_nodes())
#print(cg.CN_graph.number_of_edges())