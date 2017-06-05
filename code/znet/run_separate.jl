using HDF5
include("codegen.jl")
using CodeGen
import JSON
net_path     = ARGS[1]
weights_path = ARGS[2]
BATCH_SIZE   = parse(Int, ARGS[3])
iter_number  = ARGS[4] 
SIMD_WIDTH = 16

function updtensor!(tensors, t, dim)
   if haskey(tensors, t)
      #just do this work
      updparam!(tensors[t], dim)
   else
      tensors[t] = Tensor(dim)
   end
end

function getpoolparam(l)
	0
end
 
function getconvparam(l)
  return l 
end

function getdeconvparam(l)
  return l 
end

function  getconvdim(bot, param)
	top_dim = [-1, -1, -1, -1 ,-1]
	top_dim[1] = bot.dim[1]
	top_dim[2] = (param["num_output"]) 
    top_dim[2] = convert(Int32, ceil(top_dim[2] / S) * S)

	for i in [3,4,5]
		top_dim[i] = (bot.dim[i] - param["kernel_size"][i - 2] + 
							2*param["pad"][i - 2])/param["stride"][i - 2] + 1
	end
   return top_dim 
end

function  getpooldim(bot, param)
	top_dim = [-1, -1, -1, -1 ,-1]
	top_dim[1] = bot.dim[1]
   top_dim[2] = bot.dim[2] 
	for i in [3,4,5]
		top_dim[i] = ceil(bot.dim[i] / 2)
	end
	return top_dim
end

function getdeconvweights(data_file, l)
   [0]
end

function getconvweights(data_file, l)
   [0]
end

type Tensor
    size::Int64
	dim::Array{Int32}

	function Tensor(dim)
		this = new()
		this.dim = dim
        this.size = 1
        #println("Start for $(dim)...")
        #print(1)
        for d in this.dim
            #print("* $(d)")
            this.size *= d
        end
        #println()
        #println("Result for $(dim): $(this.size)")

		return this
	end
end

abstract Layer

type ZnnPhiDeconv <: Layer
   name::String
   top::String
   bot::String
   weights::Array{Int}
   kernel_size::Int
   bias_size::Int
   forward::Function


   function ZnnPhiDeconv(name::String, params::Dict, weights::Array, bot::String, top::String)
      this = new()
      
      this.name = name 
      this.bot = bot
      this.top = top
      this.weights = weights
      this.kernel_size = params["kd"] * 
                         params["khw"] * 
                         params["khw"] * 
                         params["ofm"] * 
                         params["ifm"]  
      this.bias_size = params["ofm"] 
      return this
   end
end


type ZnnPhiConv <: Layer
   name::String
   top::String
   bot::String
   weights::Array{Int}
   kernel_size::Int
   bias_size::Int
   params::Dict

   forward::Function

   function ZnnPhiConv(name::String, params::Dict, weights::Array, bot::String, top::String)
      this = new()
      this.params = params 
      this.name = name 
      this.bot = bot
      this.top = top
      this.weights     = weights
      this.kernel_size = params["kd"] * 
                         params["khw"] * 
                         params["khw"] * 
                         params["ofm"] * 
                         params["ifm"]  
      this.bias_size = params["ofm"] 
      return this
   end
end

function updparam!(tensor, dim)
   size = 1
   for d in dim
       size *= d
   end
   tensor.size = max(tensor.size, size)
end

data_file = h5open(weights_path, "r")
net = JSON.parsefile(net_path) 
layers_json = net["layer"]

S = SIMD_WIDTH 

tensors     = Dict()
layer_info  = Dict() 
layers      = [] 

for l in layers_json
   t = l["type"]
   
   if t == "Input" 
      dim = l["input_param"]["shape"][1]["dim"]
      dim[1] = BATCH_SIZE
      dim[2] = convert(Int32, ceil(dim[2] / S) * S)

      tensors[l["name"]] = Tensor(dim)
      #bot = tensors["bottom"]]
   elseif t == "Convolution"
      conv_param = getconvparam(l["convolution_param"])
      bot = tensors[l["bottom"][1]]
      top_dim = getconvdim(bot, conv_param)
      updtensor!(tensors, l["top"][1], top_dim)

      conv_weights = getconvweights(data_file, l)
   
      params = Dict()
      params["bn"]   = bot.dim[1]
      params["ifm"] = bot.dim[2]
      params["ofm"] = top_dim[2] 
      params["id"]  = bot.dim[3]
      params["ihw"] = bot.dim[4]

      params["kd"]    = conv_param["kernel_size"][1]
      params["khw"]   = conv_param["kernel_size"][2]
      params["padd"]  = conv_param["pad"][1]
      params["padhw"] = conv_param["pad"][2]

	  #println("$(b), $(ifm), $(ofm), $(id), $(ihw), $(kd), $(khw), $(padd), $(padhw)")
      name = l["name"]
      layer = ZnnPhiConv(name, params, conv_weights, l["bottom"][1], l["top"][1])
      layer_info[l["name"]] = layer
      push!(layers, l["name"])
   elseif t == "ELU" 
   elseif t == "BatchNorm" 
   elseif t == "Scale" 
   elseif t == "Eltwise" 
      bot = tensors[l["bottom"][1]]
      top_dim = bot.dim
      updtensor!(tensors, l["top"][1], top_dim)
   elseif t == "Pooling" 
      pool_param = getpoolparam(l["pooling_param"])
      bot = tensors[l["bottom"][1]]
      top_dim = getpooldim(bot, pool_param)
      updtensor!(tensors, l["top"][1], top_dim)
      #=layer = PoolLayer(pool_param, bot, tensors[l["top"][1]]) 
      push!(layers, layer)=#
   elseif t == "Deconvolution" 
      bot = tensors[l["bottom"][1]]
      top_dim = copy(bot.dim)
      top_dim[4] *= 2;
      top_dim[5] *= 2;
      updtensor!(tensors, l["top"][1], top_dim)

      deconv_param = getdeconvparam(l["convolution_param"]) 	
      deconv_weights = getdeconvweights(data_file, l)
   
      params = Dict()
      S = SIMD_WIDTH 
      params["bn"]   = bot.dim[1]
      params["ifm"] = bot.dim[2]
      params["ofm"] = top_dim[2]
      params["id"]  = bot.dim[3]
      params["ihw"] = bot.dim[4]

      params["kd"]    = deconv_param["kernel_size"][1]
      params["khw"]   = deconv_param["kernel_size"][2]
      params["padd"]  = deconv_param["pad"][1]
      params["padhw"] = deconv_param["pad"][2]
      
      name = l["name"]
      #layer = ZnnPhiDeconv(name, params, deconv_weights, l["bottom"][1], l["top"][1])
      #push!(layers, layer)
   elseif t == "Sigmoid" 
   elseif t == "" 
   else println(t)
   end
end

lcount = 1 
for l_name in layers
   l = layer_info[l_name]
   if isa(l, ZnnPhiConv) || isa(l, ZnnPhiDeconv)
      f = open("./cpp_out/layer_$(lcount)_b$(BATCH_SIZE).cpp", "w")
      lcount += 1

      write(f, "#include <znn/interface/conv_wrapper.hpp>\n")
      write(f, "#include <znn/tensor/tensor.hpp>\n")
      write(f, "#include <iostream>\n")
      write(f, "#include <chrono>\n")
      write(f, "#include <omp.h>\n")
      write(f, "\n")
      write(f, "\n")
      write(f, "int main(void) {\n")

      for (k, t) in tensors
         write(f, "    znn::phi::hbw_array<float> tensor_$(k)($(t.size));\n")
      end

      write(f, "    znn::phi::hbw_array<float> tensor_$(l.name)_kernel($(l.kernel_size));\n")
      write(f, "    znn::phi::hbw_array<float> tensor_$(l.name)_bias($(l.bias_size));\n")

      conv_num = 1
      write(f, "\n")
      write(f, "   std::vector<znn::phi::ConvWrapper> znnphi_convs($(conv_num));\n")
      write(f, "\n")
      write(f, "   int conv_params[$(conv_num)][9];\n")

      bn    = l.params["bn"]
      ifm   = l.params["ifm"]
      ofm   = l.params["ofm"]
      id    = l.params["id"]
      ihw   = l.params["ihw"]
      kd    = l.params["kd"]
      khw   = l.params["khw"]
      padd  = l.params["padd"]
      padhw = l.params["padhw"]
      count = 0
      write(f, "   conv_params[$(count)][0] = $(bn);\n")
      write(f, "   conv_params[$(count)][1] = $(ifm);\n")
      write(f, "   conv_params[$(count)][2] = $(ofm);\n")
      write(f, "   conv_params[$(count)][3] = $(id);\n")
      write(f, "   conv_params[$(count)][4] = $(ihw);\n")
      write(f, "   conv_params[$(count)][5] = $(kd);\n")
      write(f, "   conv_params[$(count)][6] = $(khw);\n")
      write(f, "   conv_params[$(count)][7] = $(padd);\n")
      write(f, "   conv_params[$(count)][8] = $(padhw);\n")

      write(f, "   for (int i = 0; i < $(conv_num); i++) {\n")
      write(f, "      znnphi_convs[i].init(conv_params[i][0], conv_params[i][1], conv_params[i][2],\n")
      write(f, "                           conv_params[i][3], conv_params[i][4], conv_params[i][5],\n")
      write(f, "                           conv_params[i][6], conv_params[i][7], conv_params[i][8]);\n")
      write(f, "   } // for (0..$(conv_num))\n")

      write(f, "   std::cout << \"Starting the run...\\n\";\n")
      TIME_EACH_LAYER=true
      buffer = IOBuffer() 
      conv_count = 0
      if isa(l, ZnnPhiConv)
         run = ("    znnphi_convs[$(conv_count)].forward(tensor_$(l.bot).data(), \n\n") * 
               ("                           tensor_$(l.top).data(), \n\n") * 
               ("                           tensor_$(l.name)_kernel.data(), \n\n") * 
               ("                           tensor_$(l.name)_bias.data());\n\n")	
         conv_count += 1
         if TIME_EACH_LAYER
            print(buffer, timeit(run, iter_number))
         else
            print(buffer, run)
         end
      end
   end

   TIME_TOTAL=true
   if TIME_TOTAL
      print(buffer, timeit(takebuf_string(buffer), iter_number, "total: "))
   end
   write(f, takebuf_string(buffer))
   write(f, "} //int main\n")
end
