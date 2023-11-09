import onnx
from networkx import DiGraph

class OnnxParser:
    '''
    onnx parser.
    layerout: NCHW
    '''
    
    def __init__(
            self,
            model: onnx.ModelProto
        ) -> None:
        self.graph = model.graph
        self.layers = []
        for layer in self.graph.node :
            layer_info = {
                'name': layer.name,
                'op_type': layer.op_type,
                'inputs': layer.input,
                'outputs': layer.output,
                'input_shapes': [self.get_tensor_shape(input) for input in layer.input],
                'output_shapes': [self.get_tensor_shape(output) for output in layer.output]
            }
            if layer.op_type == 'Conv':
                layer_info['conv_attributes'] = self.parse_conv_attributes(layer)
            elif layer.op_type == 'MaxPool':
                layer_info['maxpool_attributes'] = self.parse_maxpool_attributes(layer)
            elif layer.op_type == 'Gemm':
                layer_info['attributes'] = self.parse_gemm_attributes(layer)
            elif layer.op_type == 'BatchNormalization':
                layer_info['batchnorm_attributes'] = self.parse_batchnorm_attributes(layer)
            self.layers.append(layer_info)

        self.layers_dict = {layer['name']: layer for layer in self.layers}

        for layer in self.layers:
            if layer['input_shapes'][0] is None:
                self.infer_input_shape(layer['name'])
            if layer['output_shapes'][0] is None:
                self.infer_output_shape(layer['name'])

        self.layer_graph = self.build_graph()

        isolated_nodes = [node for node in self.layer_graph.nodes() if self.layer_graph.in_degree(node) == 0 and self.layer_graph.out_degree(node) == 0]
        if isolated_nodes:
            raise ValueError(f'there are isolated layers:{isolated_nodes}.')
        else:
            print("onnx model parse completed!")

    def show(self):
        for layer in self.layers :
            print(layer,'\n')

    def parse_batchnorm_attributes(self, layer):
        '''提取BatchNormalization层的特定属性'''
        attributes = {
            'epsilon': 1e-5,  # 默认值
            'momentum': 0.9,  # 默认值
        }

        # 遍历属性以获取epsilon和momentum
        for attr in layer.attribute:
            if attr.name == 'epsilon':
                attributes['epsilon'] = attr.f
            elif attr.name == 'momentum':
                attributes['momentum'] = attr.f

        # 假设scale, B, mean, var是通过inputs来提供的
        # scale, B, mean, var对应的是layer.input[1], layer.input[2], layer.input[3], layer.input[4]
        if len(layer.input) > 1:
            scale_name = layer.input[1]
            attributes['scale'] = self.get_tensor_shape(scale_name)
        if len(layer.input) > 2:
            B_name = layer.input[2]
            attributes['B'] = self.get_tensor_shape(B_name)
        if len(layer.input) > 3:
            mean_name = layer.input[3]
            attributes['mean'] = self.get_tensor_shape(mean_name)
        if len(layer.input) > 4:
            var_name = layer.input[4]
            attributes['var'] = self.get_tensor_shape(var_name)

        return attributes

    def parse_conv_attributes(self, layer):
        '''提取卷积层的特定属性'''
        attributes = {}
        for attr in layer.attribute:
            # 检查卷积核形状
            if attr.name == 'kernel_shape':
                attributes['kernel_shape'] = attr.ints
            # 检查步长
            elif attr.name == 'strides':
                attributes['strides'] = attr.ints
            # 检查填充
            elif attr.name == 'pads':
                attributes['pads'] = attr.ints

        weights_name = layer.input[1]  #卷积层的第二个输入是权重
        weights_shape = self.get_tensor_shape(weights_name)
        if weights_shape is not None:
            attributes['output_channels'] = weights_shape[0]  #权重形状的第一个维度是输出通道数
            attributes['input_channels'] = weights_shape[1]  #权重形状的第二个维度是输入通道数

        #检查是否有 bias
        attributes['has_bias'] = len(layer.input) == 3  #如果输入数量为3，则存在bias

        return attributes

    def parse_maxpool_attributes(self, layer):
        '''提取MaxPool层的特定属性'''

        attributes = {}
        for attr in layer.attribute:
            if attr.name == 'kernel_shape':
                attributes['kernel_shape'] = attr.ints
            elif attr.name == 'strides':
                attributes['strides'] = attr.ints
            elif attr.name == 'pads':
                attributes['pads'] = attr.ints

        return attributes
    
    def get_tensor_shape(self, tensor_name):
        '''通过tensor名称查找其形状'''

        for value_info in self.graph.input:  #模型的实际输入和常量输入
            if value_info.name == tensor_name:
                return self.get_shape_from_value_info(value_info)
        for value_info in self.graph.value_info:  #中间张量的形状信息
            if value_info.name == tensor_name:
                return self.get_shape_from_value_info(value_info)
        for value_info in self.graph.output:  #模型的输出
            if value_info.name == tensor_name:
                return self.get_shape_from_value_info(value_info)
        
        '''通过initializer名称查找其形状'''
        for initializer in self.graph.initializer:
            if initializer.name == tensor_name:
                return [dim for dim in initializer.dims]

        return None

    def get_shape_from_value_info(self, value_info):
        '''从ValueInfoProto提取形状'''
        try:
            shape = [dim.dim_value for dim in value_info.type.tensor_type.shape.dim]
            if shape[0] == 0:
                shape[0] = 1
            return shape
        except:
            return None  #如果形状信息不完整或不存在，返回None
        
    def parse_gemm_attributes(self, layer):
        '''提取Gemm层的特定属性'''
        attributes = {}
        for attr in layer.attribute:
            if attr.name == 'alpha':
                attributes['alpha'] = attr.f
            elif attr.name == 'beta':
                attributes['beta'] = attr.f
            elif attr.name == 'transA':
                attributes['transA'] = attr.i
            elif attr.name == 'transB':
                attributes['transB'] = attr.i

        #提取权重矩阵的形状
        weights_name = layer.input[1]  #假设第二个输入是权重矩阵
        weights_shape = self.get_tensor_shape(weights_name)
        attributes['weights_shape'] = weights_shape

        #检查并提取偏置的形状
        if len(layer.input) > 2:  #如果存在第三个输入，那么它是偏置
            bias_name = layer.input[2]
            bias_shape = self.get_tensor_shape(bias_name)
            attributes['has_bias'] = True
            attributes['bias_shape'] = bias_shape

        return attributes
    
    def build_graph(self):
        '''创建有向图'''
        graph = DiGraph()

        #添加节点
        for layer in self.layers:
            graph.add_node(layer['name'], op_type=layer['op_type'])

        #添加边
        for layer in self.layers:
            #对于每个输出，查找下一个层的输入，并创建一条边
            for output in layer['outputs']:
                #查找所有使用此输出作为输入的层
                next_layers = self.find_next_layers(output)
                for next_layer in next_layers:
                    graph.add_edge(layer['name'], next_layer['name'])

        return graph

    def find_next_layers(self, tensor_name):
        # 查找下一个层（tensor_name作为输入的层）
        next_layers = []
        for layer in self.layers:
            if tensor_name in layer['inputs']:
                next_layers.append(layer)
        return next_layers
    
    def find_pre_layer(self, tensor_name):
        # 查找上一个层（tensor_name作为输入的层）
        for layer in self.layers:
            if tensor_name in layer['outputs']:
                return layer
            
        return None
    
    def infer_input_shape(self, layer_name):
        '''通过上一层的输出来推断这一层的输入形状'''
        layer = self.layers_dict[layer_name]
        if layer['input_shapes'][0] is None:
            # Find the previous layer based on the input tensor name
            pre_layer = self.find_pre_layer(layer['inputs'][0])
            # Ensure the previous layer is found
            if pre_layer is not None:
                layer['input_shapes'][0] = self.infer_output_shape(pre_layer['name'])
                if layer['op_type'] == 'Add':
                    layer['input_shapes'][1] = layer['input_shapes'][0]
            else:
                raise ValueError(f"Previous layer for {layer_name} not found")
        return layer['input_shapes'][0]

    def infer_output_shape(self, layer_name):
        '''通过这一层的输入来推断这一层的输出形状'''
        layer = self.layers_dict[layer_name]
        input_shape = layer['input_shapes'][0]
        if input_shape is None:
            raise ValueError(f"Input shape for layer '{layer_name}' is None, which indicates missing shape information.")
        if layer['output_shapes'][0] is None:
            if input_shape == None:#先推断输入
                input_shape = self.infer_input_shape(layer_name)

            if layer['op_type'] == 'Relu':
                layer['output_shapes'][0] = input_shape
            if layer['op_type'] == 'Add':
                layer['output_shapes'][0] = input_shape
            if layer['op_type'] == 'BatchNormalization':
                layer['output_shapes'][0] = input_shape
            if layer['op_type'] == 'Flatten':
                layer['output_shapes'][0] = [input_shape[0], input_shape[1]*input_shape[2]*input_shape[3]]
            if layer['op_type'] == 'GlobalAveragePool':
                layer['output_shapes'][0] = [input_shape[0], input_shape[1], 1, 1]
            if layer['op_type'] == 'Gemm':
                weights_shape = layer['attributes']['weights_shape']
                if layer['attributes']['transB'] == 1:
                    if layer['attributes']['transA'] == 0:
                        layer['output_shapes'][0] = [input_shape[0], weights_shape[0]]
                    else:
                        layer['output_shapes'][0] = [input_shape[1], weights_shape[0]]
                else :
                    if layer['attributes']['transA'] == 0:
                        layer['output_shapes'][0] = [input_shape[0], weights_shape[1]]
                    else:
                        layer['output_shapes'][0] = [input_shape[1], weights_shape[1]]
            if layer['op_type'] == 'MaxPool':
                N, C, H, W = input_shape
                pad_h_top, pad_w_left, pad_h_bottom, pad_w_right = layer['maxpool_attributes']['pads']
                kernel_h, kernel_w = layer['maxpool_attributes']['kernel_shape']
                stride_h, stride_w = layer['maxpool_attributes']['strides']
                H_out = ((H + pad_h_top + pad_h_bottom - kernel_h) // stride_h) + 1
                W_out = ((W + pad_w_left + pad_w_right - kernel_w) // stride_w) + 1
                layer['output_shapes'][0] = [N, C, H_out, W_out]
            if layer['op_type'] == 'Conv':
                N, C, H, W = input_shape
                pad_h_top, pad_w_left, pad_h_bottom, pad_w_right = layer['conv_attributes']['pads']
                kernel_h, kernel_w = layer['conv_attributes']['kernel_shape']
                stride_h, stride_w = layer['conv_attributes']['strides']
                total_pad_h = pad_h_top + pad_h_bottom
                total_pad_w = pad_w_left + pad_w_right
                H_out = (H + total_pad_h - kernel_h) // stride_h + 1
                W_out = (W + total_pad_w - kernel_w) // stride_w + 1
                C_out = layer['conv_attributes']['output_channels']
                layer['output_shapes'][0] = [N, C_out, H_out, W_out]
        return layer['output_shapes'][0]
