require PolyHok
#PolyHok.defmodule SkeKernels do
#
# defk map_step_2_para_no_resp_kernel(d_array,  step, par1, par2,size,f) do
#    globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
#    id  = step * globalId
#    #f(id,id)
#    if (globalId < size) do
#      f(d_array+id,par1,par2)
#    end
#  end
#
#end

PolyHok.defmodule Ske do
  #defmacro __using__(_opts) do
  #     IO.puts "You are USIng!"
  #    end

  include CAS_Poly
  #include CAS_Double

## -----------------------------------------------------------------------------------------------------------------------------------------------------
## REDUCE-----------------------------------------------------------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------------------------------------------------------------

  # def reduce(ref, initial, f) do
  #   #IO.inspect(PolyHok.get_gnx(ref))
  #   shape = PolyHok.get_shape_gnx(ref)
  #   type = PolyHok.get_type_gnx(ref)
  #   size = Tuple.product(shape)
  #   result_gpu  = PolyHok.new_gnx(Nx.tensor([[initial]] , type: type))

  #   threadsPerBlock = 256
  #   blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
  #   numberOfBlocks = blocksPerGrid

  #   #cudaDeviceProp prop
  #   #cudaGetDeviceProperties(&prop, 0)
  #   #blocks = prop.multiProcessorCount*2
  #   #threads = 256

  #   #PolyHok.spawn(&Ske.reduce_kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref, result_gpu, initial, size, f, size])
  #   case type do
  #     {:f,32} -> cas = PolyHok.phok (fn (x,y,z) -> cas_float(x,y,z) end)
  #             PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])
  #             #PolyHok.spawn(&Ske.reduce_kernel_nvidia_k5/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, size, cas, f])

  #     {:f,64} -> cas = PolyHok.phok (fn (x,y,z) -> cas_double(x,y,z) end)
  #             PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])
  #             #PolyHok.spawn(&Ske.reduce_kernel_nvidia_k5/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, size, cas, f])

  #     {:s,32} -> cas = PolyHok.phok (fn (x,y,z) -> cas_int(x,y,z) end)
  #             PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])
  #             #PolyHok.spawn(&Ske.reduce_kernel_nvidia_k5/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, size, cas, f])

  #     x -> raise "new_gnx: type #{x} not suported"
  #   end

  #   result_gpu
  # end

  def reduce(ref, initial, f, sel_ver \\01) do
    #IO.inspect(PolyHok.get_gnx(ref))
    shape = PolyHok.get_shape_gnx(ref)
    type = PolyHok.get_type_gnx(ref)
    size = Tuple.product(shape)
    result_gpu  = PolyHok.new_gnx(Nx.tensor([[initial]] , type: type))

    threadsPerBlock = 256
    blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
    numberOfBlocks = blocksPerGrid

    #cudaDeviceProp prop
    #cudaGetDeviceProperties(&prop, 0)
    #blocks = prop.multiProcessorCount*2
    #threads = 256

    #PolyHok.spawn(&Ske.reduce_kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref, result_gpu, initial, size, f, size])
    cas = case type do
      {:f,32} -> PolyHok.phok (fn (x,y,z) -> cas_float(x,y,z) end)

      {:f,64} -> PolyHok.phok (fn (x,y,z) -> cas_double(x,y,z) end)

      {:s,32} -> PolyHok.phok (fn (x,y,z) -> cas_int(x,y,z) end)

      x -> raise "new_gnx: type #{x} not suported"
    end

    case sel_ver do
      01 -> PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])

      02 -> PolyHok.spawn(&Ske.reduce_kernel_nvidia_k4/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, size, cas, f])

      03 -> PolyHok.spawn(&Ske.reduce_kernel_nvidia_k5/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, size, cas, f])

      x  -> raise "reduce implementation #{x} not available"
    end

    result_gpu
  end

## REDUCE CLÁSSICO => SEL_VER : 01
  defk reduce_kernel(a, ref4, initial, n, cas, f) do
  #defk reduce_kernel(a, ref4, initial, n, f) do
    __shared__ cache[256]

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x

    temp = initial

    while (tid < n) do
      temp = f(a[tid], temp)
      tid = blockDim.x * gridDim.x + tid
    end

    cache[cacheIndex] = temp
      __syncthreads()

    i = blockDim.x/2

    while (i != 0 ) do  ###&& tid < n) do
      #tid = blockDim.x * gridDim.x + tid
      if (cacheIndex < i) do
        cache[cacheIndex] = f(cache[cacheIndex + i] , cache[cacheIndex])
      end

      __syncthreads()
      i = i/2
    end

    if (cacheIndex == 0) do
      current_value = ref4[0]
      #while(!(current_value == atomic_cas(ref4,current_value,f(cache[0],current_value)))) do
      while(!(current_value == cas(ref4,current_value,f(cache[0],current_value)))) do
        current_value = ref4[0]
      end
    end
  end

## REDUCE NVIDIA OTIMIZAÇÃO 4 => SEL_VER : 02
  defk reduce_kernel_nvidia_k4(a, ref4, n, cas, f) do
    printf("TESTE4")
    __shared__ cache[256]

    unsigned int tid
    tid = blockIdx.x*(blockDim.x) + threadIdx.x;
    unsigned int cacheIndex
    cacheIndex = threadIdx.x;

    cache[cacheIndex] = a[tid] + a[tid+blockDim.x]
    __syncthreads();

    for (unsigned int s = blockDim.x; s > 0; s = s/2) do
      if (cacheIndex < s) do
        cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + s]
      end
      __syncthreads();
    end

    if (cacheIndex == 0) do
      ref4[blockIdx.x] = cache[0];
    end
  end

## REDUCE NVIDIA OTIMIZAÇÃO 5 => SEL_VER : 03
  defk reduce_kernel_nvidia_k5(a, ref4, n, cas, f) do
    printf("TESTE5")
    __shared__ cache[256]

    tid = blockIdx.x*(blockDim.x) + threadIdx.x;
    cacheIndex = threadIdx.x;

    #tem que usar o f pra ele identificar o tipo do f
    #cache[cacheIndex] = a[tid] + a[tid+blockDim.x]
    cache[cacheIndex] = f(a[tid],a[tid+blockDim.x])
    __syncthreads();

    #syntaxe do for tem que arrumar
    for s in range(blockDim.x, 32, s = s/2) do
      if (cacheIndex < s) do
        cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + s]
      end
      __syncthreads();
    end

    if (cacheIndex < 32) do
      cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + 32]
      cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + 16]
      cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + 8]
      cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + 4]
      cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + 2]
      cache[cacheIndex] = cache[cacheIndex] + cache[cacheIndex + 1]
    end

    if (cacheIndex == 0) do
      #printf("%d %d %d %d %d \\n",cache[0],cache[1],cache[2],cache[3],cache[4])
      ref4[blockIdx.x] = cache[0]
    end
  end

## -----------------------------------------------------------------------------------------------------------------------------------------------------
## MAP--------------------------------------------------------------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------------------------------------------------------------

# ADICIONAR: Parâmetro ID, para identificar se valor passado pra função será ponteiro da primeira posição ou ponteiro da posição atual

## SELECT : 1 GNX : 0 PARA
  @defaults %{coord: false, return: true, dim: :one}
  def map(a,b,c \\[],options \\[])
  def map({:nx, type, shape, name , ref}, func, [], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map_0para_1D({:nx, type, shape, name, ref}, func)

              else if (not coord && return) do
                map_0para_1D_resp({:nx, type, shape, name, ref}, func)

              else if (coord && not return) do
                map_0para_coord_1D({:nx, type, shape, name, ref}, func)

              else if (coord && return) do
                map_0para_coord_1D_resp({:nx, type, shape, name, ref}, func)
              end
              end
              end
              end

     :two ->  if (not coord && not return) do
                map_0para_2D({:nx, type, shape, name, ref}, func)

              else if (not coord && return) do
                map_0para_2D_resp({:nx, type, shape, name, ref}, func)

              else if (coord && not return) do
                map_0para_coord_2D({:nx, type, shape, name, ref},  func)

              else if (coord && return) do
                map_0para_coord_2D_resp({:nx, type, shape, name, ref}, func)
              end
              end
              end
              end

  end
  end

## SELECT : 1 GNX : 1 PARA
  def map({:nx, type, shape, name , ref}, func, [par1], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map_1para_1D({:nx, type, shape, name, ref}, par1, func)

              else if (not coord && return) do
                map_1para_1D_resp({:nx, type, shape, name , ref}, par1, func)

              else if (coord && not return) do
                map_1para_coord_1D({:nx, type, shape, name, ref}, par1, func)

              else if (coord && return) do
                map_1para_coord_1D_resp({:nx, type, shape, name, ref}, par1, func)
              end
              end
              end
              end

     :two ->  if (not coord && not return) do
                map_1para_2D({:nx, type, shape, name, ref}, par1, func)

              else if (not coord && return) do
                map_1para_2D_resp({:nx, type, shape, name, ref}, par1, func)

              else if (coord && not return) do
                map_1para_coord_2D({:nx, type, shape, name, ref}, par1, func)

              else if (coord && return) do
                map_1para_coord_2D_resp({:nx, type, shape, name, ref}, par1, func)
              end
              end
              end
              end

  end
  end

## SELECT : 1 GNX : 2 PARA
  def map({:nx, type, shape, name , ref}, func, [par1,par2], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map_2para_1D({:nx, type, shape, name, ref}, par1, par2, func)

              else if (not coord && return) do
                map_2para_1D_resp({:nx, type, shape, name, ref}, par1, par2, func)

              else if (coord && not return) do
                map_2para_coord_1D({:nx, type, shape, name, ref}, par1, par2, func)

              else if (coord && return) do
                map_2para_coord_1D_resp({:nx, type, shape, name, ref}, par1, par2, func)
              end
              end
              end
              end

     :two ->  if (not coord && not return) do
                map_2para_2D({:nx, type, shape, name, ref}, par1, par2, func)

              else if (not coord && return) do
                map_2para_2D_resp({:nx, type, shape, name, ref}, par1, par2, func)

              else if (coord && not return) do
                map_2para_coord_2D({:nx, type, shape, name, ref}, par1, par2, func)

              else if (coord && return) do
                map_2para_coord_2D_resp({:nx, type, shape, name, ref}, par1, par2, func)
              end
              end
              end
              end

  end
  end

## SELECT : 1 GNX : 3 PARA
  def map({:nx, type, shape, name , ref}, func, [par1,par2,par3], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map_3para_1D({:nx, type, shape, name, ref}, par1, par2, par3, func)

              else if (not coord && return) do
                map_3para_1D_resp({:nx, type, shape, name, ref}, par1, par2, par3, func)

              else if (coord && not return) do
                map_3para_coord_1D({:nx, type, shape, name, ref}, par1, par2, par3, func)

              else if (coord && return) do
                map_3para_coord_1D_resp({:nx, type, shape, name, ref}, par1, par2, par3, func)
              end
              end
              end
              end

     :two ->  if (not coord && not return) do
                map_3para_2D({:nx, type, shape, name, ref}, par1, par2, par3, func)

              else if (not coord && return) do
                map_3para_2D_resp({:nx, type, shape, name, ref}, par1, par2, par3, func)

              else if (coord && not return) do
                map_3para_coord_2D({:nx, type, shape, name, ref}, par1, par2, par3, func)

              else if (coord && return) do
                map_3para_coord_2D_resp({:nx, type, shape, name, ref}, par1, par2, par3, func)
              end
              end
              end
              end

  end
  end

  ### \/ Para 0 parametros devo usar uma lista vazia [] ou só não usar?
  ### \/ Problema de não colocar: quando for 0 parametros mas quer passar lista de Options ele vai quebrar dizendo que tá passando parâmetros demais

## SELECT : 2 GNX : 0 PARA
  def map({:nx, type, shape, name , ref}, {:nx, type2, shape2, name2 , ref2}, func, [], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map2_0para_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func)

              else if (not coord && return) do
                map2_0para_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func)

              else if (coord && not return) do
                map2_0para_coord_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func)

              else if (coord && return) do
                map2_0para_coord_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func)
              end
              end
              end
              end

     :two ->  if (not coord && not return) do
                map2_0para_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func)

              else if (not coord && return) do
                map2_0para_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func)

              else if (coord && not return) do
                map2_0para_coord_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func)

              else if (coord && return) do
                map2_0para_coord_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func)
              end
              end
              end
              end

  end
  end

## SELECT : 2 GNX : 1 PARA
  def map({:nx, type, shape, name , ref}, {:nx, type2, shape2, name2 , ref2}, func, [par1], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
   :one ->   if (not coord && not return )do
               map2_1para_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, func)

             else if (not coord && return) do
               map2_1para_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, func)

             else if (coord && not return) do
               map2_1para_coord_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, func)

             else if (coord && return) do
               map2_1para_coord_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, func)
             end
             end
             end
             end

     :two ->  if (not coord && not return) do
               map2_1para_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, func)

              else if (not coord && return) do
               map2_1para_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, func)

              else if (coord && not return) do
                map2_1para_coord_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, func)

              else if (coord && return) do
                map2_1para_coord_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, func)
              end
              end
              end
              end

  end
  end

## SELECT : 2 GNX : 2 PARA
  def map({:nx, type, shape, name , ref}, {:nx, type2, shape2, name2 , ref2}, func, [par1, par2], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map2_2para_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, func)

              else if (not coord && return) do
                map2_2para_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, func)

              else if (coord && not return) do
                map2_2para_coord_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, func)

              else if (coord && return) do
                map2_2para_coord_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, func)
              end
              end
              end
              end

     :two ->  if (not coord && not return) do
               map2_2para_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, func)

              else if (not coord && return) do
                map2_2para_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, func)

              else if (coord && not return) do
                map2_2para_coord_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, func)

              else if (coord && return) do
                map2_2para_coord_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, func)
              end
              end
              end
              end

  end
  end

## SELECT : 2 GNX : 3 PARA
  def map({:nx, type, shape, name , ref}, {:nx, type2, shape2, name2 , ref2}, func, [par1, par2, par3], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
   :one ->   if (not coord && not return )do
               map2_3para_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, par3, func)

             else if (not coord && return) do
               map2_3para_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, par3, func)

             else if (coord && not return) do
               map2_3para_coord_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, par3, func)

             else if (coord && return) do
               map2_3para_coord_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, par3, func)
             end
             end
             end
             end

     :two ->  if (not coord && not return) do
                map2_3para_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, par3, func)

              else if (not coord && return) do
                map2_3para_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, par3, func)

              else if (coord && not return) do
                map2_3para_coord_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, par3, func)

              else if (coord && return) do
                map2_3para_coord_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, par1, par2, par3, func)
              end
              end
              end
              end

  end
  end

## SELECT : 3 GNX : 0 PARA
 def map({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func, [], options )do
   %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
 case dim do
   :one ->  if (not coord && not return )do
              map3_0para_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func)

            else if (not coord && return) do
              map3_0para_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func)

            else if (coord && not return) do
              map3_0para_coord_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func)

            else if (coord && return) do
              map3_0para_coord_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func)
            end
            end
            end
            end

  :two ->   if (not coord && not return) do
              map3_0para_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func)

            else if (not coord && return) do
              map3_0para_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func)

            else if (coord && not return) do
              map3_0para_coord_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func)

            else if (coord && return) do
              map3_0para_coord_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func)
            end
            end
            end
            end
 end
 end

## SELECT : 3 GNX : 1 PARA
  def map({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func, [par1], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                map3_1para_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, func)

              else if (not coord && return) do
                map3_1para_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, func)

              else if (coord && not return) do
                map3_1para_coord_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, func)

              else if (coord && return) do
                map3_1para_coord_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, func)
              end
              end
              end
              end

    :two ->   if (not coord && not return) do
                map3_1para_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, func)

              else if (not coord && return) do
                map3_1para_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, func)

              else if (coord && not return) do
                map3_1para_coord_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, func)

              else if (coord && return) do
                map3_1para_coord_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, func)
              end
              end
              end
              end
  end
  end

## SELECT : 3 GNX : 2 PARA
 def map({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func, [par1, par2], options )do
   %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
 case dim do
   :one ->   if (not coord && not return )do
               map3_2para_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3 , ref3}, par1, par2, func)

             else if (not coord && return) do
               map3_2para_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, func)

             else if (coord && not return) do
               map3_2para_coord_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, func)

             else if (coord && return) do
               map3_2para_coord_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, func)
             end
             end
             end
             end

  :two ->  if (not coord && not return) do
              map3_2para_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, func)

           else if (not coord && return) do
              map3_2para_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2 , ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, func)

           else if (coord && not return) do
              map3_2para_coord_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, func)

           else if (coord && return) do
              map3_2para_coord_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2 , ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, func)
           end
           end
           end
           end
 end
 end

## SELECT : 3 GNX : 3 PARA
 def map({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, func, [par1, par2, par3], options )do
   %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
 case dim do
   :one ->  if (not coord && not return )do
              map3_3para_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, par3, func)

            else if (not coord && return) do
              map3_3para_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, par3, func)

            else if (coord && not return) do
              map3_3para_coord_1D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2 , ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, par3, func)

            else if (coord && return) do
              map3_3para_coord_1D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, par3, func)
            end
            end
            end
            end

  :two ->   if (not coord && not return) do
              map3_3para_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, par3, func)

            else if (not coord && return) do
              map3_3para_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, par3, func)

            else if (coord && not return) do
              map3_3para_coord_2D({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, par3, func)

            else if (coord && return) do
              map3_3para_coord_2D_resp({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, {:nx, type3, shape3, name3, ref3}, par1, par2, par3, func)
            end
            end
            end
            end
 end
 end

## -----------------------------------------------------------------------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## 1 GNX
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## MAP = 1 GNX; 0 PARAMETERS; 1D;
  defk map_0para_1D_kernel(d_array, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      f(d_array[id])
    end
  end
  def map_0para_1D(d_array, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    #size = l*step
    size = l
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_0para_1D_kernel/4,{nBlocks,1,1},{block_size,1,1},[d_array,step,size,f])
    d_array
  end
## MAP = 1 GNX; 0 PARAMETERS; 1D, RETURN: TRUE;
  defk map_0para_1D_resp_kernel(d_array, ret, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      ret[id] = f(d_array[id])
    end
  end
  def map_0para_1D_resp(d_array, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    #size = l*step
    size = l
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_0para_1D_resp_kernel/5,{nBlocks,1,1},{block_size,1,1},[d_array,ret,step,size,f])
    ret
  end
## MAP = 1 GNX; 0 PARAMETERS; 1D, COORD: TRUE;
  defk map_0para_coord_1D_kernel(d_array, step, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array[id],idX)
    end
  end
  def map_0para_coord_1D(d_array, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    #size = sizeX*step
    size = sizeX

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_0para_coord_1D_kernel/4,{grid_rows,1,1},{block_size,block_size,1},[d_array,step,size,f])
    d_array
  end
## MAP = 1 GNX; 0 PARAMETERS; 1D, COORD: TRUE, RETURN: TRUE;
  defk map_0para_coord_1D_resp_kernel(d_array, ret, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array[id],idX)
    end
  end
  def map_0para_coord_1D_resp(d_array, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    #size = sizeX*step
    size = sizeX

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_0para_coord_1D_resp_kernel/5,{grid_rows,1,1},{block_size,block_size,1},[d_array,ret,step,size,f])
    ret
  end
## MAP = 1 GNX; 0 PARAMETERS; 2D;
  defk map_0para_2D_kernel(d_array, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      id = stride*step
      f(d_array[id])
    end
  end
  def map_0para_2D(d_array, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_0para_2D_kernel/5,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,sizeX,sizeY,f])
    d_array
  end
## MAP = 1 GNX; 0 PARAMETERS; 2D, RETURN: TRUE;
  defk map_0para_2D_resp_kernel(d_array, ret, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      ret[id] = f(d_array[id])
    end
  end
  def map_0para_2D_resp(d_array, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_0para_2D_resp_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,step,sizeX,sizeY,f])
    ret
  end
## MAP = 1 GNX; 0 PARAMETERS; 2D, COORD: TRUE;
  defk map_0para_coord_2D_kernel(d_array, step, sizeX, sizeY, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    idY = threadIdx.y + blockIdx.y * blockDim.y
    stride = idX + idY * blockDim.x * gridDim.x

    id  = step * stride
    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY

      f(d_array+id,x,y)
    end
  end
  def map_0para_coord_2D(d_array, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_0para_coord_2D_kernel/5,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,sizeX,sizeY,f])
    #PolyHok.spawn(&Ske.map_0para_coord_2D_kernel/5,{sizeX,sizeX,1},{1,1,1},[d_array,step,par1,sizeX,sizeY,f])
    d_array
  end
## MAP = 1 GNX; 0 PARAMETERS; 2D, COORD: TRUE, RETURN: TRUE;
  defk map_0para_coord_2D_resp_kernel(d_array, ret, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id  = step * stride

      ret[id] = f(d_array[id], x, y)
    end
  end
  def map_0para_coord_2D_resp(d_array, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_0para_coord_2D_resp_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,step,sizeX,sizeY,f])
    ret
  end

## MAP = 1 GNX; 1 PARAMETER; 1D;
  defk map_1para_1D_kernel(d_array, par1, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      f(d_array[id], par1)
    end
  end
  def map_1para_1D(d_array, par1, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    #size = l*step
    size = l
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_1para_1D_kernel/5,{nBlocks,1,1},{block_size,1,1},[d_array,par1,step,size,f])
    d_array
  end
## MAP = 1 GNX; 1 PARAMETER; 1D, RETURN: TRUE;
  defk map_1para_1D_resp_kernel(d_array, ret, par1, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      ret[id] = f(d_array[id], par1)
    end
  end
  def map_1para_1D_resp(d_array, par1, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    #size = l*step
    size = l
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_1para_1D_resp_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array,ret,par1,step,size,f])
    ret
  end
## MAP = 1 GNX; 1 PARAMETER; 1D, COORD: TRUE;
  defk map_1para_coord_1D_kernel(d_array, par1, step, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array[id], par1,idX)
    end
  end
  def map_1para_coord_1D(d_array, par1, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    #size = sizeX*step
    size = sizeX

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_1para_coord_1D_kernel/5,{grid_rows,1,1},{block_size,block_size,1},[d_array,par1,step,size,f])
    d_array
  end
## MAP = 1 GNX; 1 PARAMETER; 1D, COORD: TRUE, RETURN: TRUE;
  defk map_1para_coord_1D_resp_kernel(d_array, ret, par1, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array[id],par1,idX)
    end
  end
  def map_1para_coord_1D_resp(d_array, par1, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    #size = sizeX*step
    size = sizeX

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_1para_coord_1D_resp_kernel/6,{grid_rows,1,1},{block_size,block_size,1},[d_array,ret,par1,step,size,f])
    ret
  end
## MAP = 1 GNX; 1 PARAMETER; 2D;
  defk map_1para_2D_kernel(d_array, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      id = stride*step
      f(d_array[id], par1)
    end
  end
  def map_1para_2D(d_array, par1, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_1para_2D_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,par1,step,sizeX,sizeY,f])
    d_array
  end
## MAP = 1 GNX; 1 PARAMETER; 2D, RETURN: TRUE;
  defk map_1para_2D_resp_kernel(d_array, ret, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      ret[id] = f(d_array[id], par1)
    end
  end
  def map_1para_2D_resp(d_array, par1, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_1para_2D_resp_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,step,sizeX,sizeY,f])
    ret
  end
## MAP = 1 GNX; 1 PARAMETER; 2D, COORD: TRUE;
  defk map_1para_coord_2D_kernel(d_array, step, par1, sizeX, sizeY, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    idY = threadIdx.y + blockIdx.y * blockDim.y
    stride = idX + idY * blockDim.x * gridDim.x

    id  = step * stride
    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY

      f(d_array+id,par1,x,y)
    end
  end
  def map_1para_coord_2D(d_array, par1, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_1para_coord_2D_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,sizeX,sizeY,f])
    #PolyHok.spawn(&Ske.map_1para_coord_2D_kernel/6,{sizeX,sizeX,1},{1,1,1},[d_array,step,par1,sizeX,sizeY,f])
    d_array
  end
## MAP = 1 GNX; 1 PARAMETER; 2D, COORD: TRUE, RETURN: TRUE;
  defk map_1para_coord_2D_resp_kernel(d_array, ret, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id  = step * stride

      ret[id] = f(d_array[id], par1, x, y)
    end
  end
  def map_1para_coord_2D_resp(d_array, par1, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_1para_coord_2D_resp_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,step,sizeX,sizeY,f])
    ret
  end

## MAP = 1 GNX; 2 PARAMETERS; 1D;
  defk map_2para_1D_kernel(d_array, step, par1, par2, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array+id,par1,par2)
    end
  end
  def map_2para_1D(d_array, par1, par2, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    #size = l*step
    size = l
    #IO.puts(size)
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_2para_1D_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array,step,par1,par2,size,f])
    d_array
  end
## MAP = 1 GNX; 2 PARAMETERS; 1D, RETURN: TRUE;
  defk map_2para_1D_resp_kernel(d_array, ret, par1, par2, step, size, f) do
    #printf("AQUI\\n")
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      ret[id] = f(d_array[id],par1,par2)
    end
  end
  def map_2para_1D_resp(d_array, par1, par2, f) do
    block_size =  128;
    {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    #size = l*step
    size = l
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_2para_1D_resp_kernel/7,{nBlocks,1,1},{block_size,1,1},[d_array,ret,par1,par2,step,size,f])
    ret
  end
## MAP = 1 GNX; 2 PARAMETERS; 1D, COORD: TRUE;
  defk map_2para_coord_1D_kernel(d_array, par1, par2, step, size, f) do
    x = threadIdx.x + blockIdx.x * blockDim.x
    offset = x * blockDim.x * gridDim.x

    if (offset < size) do
      id = step * offset
      f(d_array[id],par1,par2,x)
    end
  end
  def map_2para_coord_1D(d_array, par1, par2, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    #size = sizeX*step
    size = sizeX

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_2para_coord_1D_kernel/6,{grid_rows,1,1},{block_size,block_size,1},[d_array,par1,par2,step,size,f])
    d_array
  end
## MAP = 1 GNX; 2 PARAMETERS; 1D, COORD: TRUE, RETURN: TRUE;
  defk map_2para_coord_1D_resp_kernel(d_array, ret, par1, par2, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    #offset = idX * blockDim.x * gridDim.x
    #if(offset < size)do
    #  ret[offset] = f(d_array[offset],par1,par2,idX)
    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array[id],par1,par2,idX)
    end
  end
  def map_2para_coord_1D_resp(d_array, par1, par2, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    #size = sizeX*step
    size = sizeX

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_2para_coord_1D_resp_kernel/7,{grid_rows,1,1},{block_size,block_size,1},[d_array,ret,par1,par2,step,size,f])
    ret
  end
## MAP = 1 GNX; 2 PARAMETERS; 2D;
  defk map_2para_2D_kernel(d_array, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      f(d_array[id], par1, par2)
    end
  end
  def map_2para_2D(d_array, par1, par2, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_2para_2D_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,par1,par2,step,sizeX,sizeY,f])
    d_array
  end
## MAP = 1 GNX; 2 PARAMETERS; 2D, RETURN: TRUE;
  defk map_2para_2D_resp_kernel(d_array, ret, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      ret[id] = f(d_array[id], par1, par2)
    end
  end
  def map_2para_2D_resp(d_array, par1, par2, f) do
    #block_size =  128;
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_2para_2D_resp_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,par2,step,sizeX,sizeY,f])
    ret
  end
## MAP = 1 GNX; 2 PARAMETERS; 2D, COORD: TRUE;
  defk map_2para_coord_2D_kernel(d_array, step, par1, par2, sizeX, sizeY, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    idY = threadIdx.y + blockIdx.y * blockDim.y
    stride = idX + idY * blockDim.x * gridDim.x

    id = stride*step
    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      ## Confirmar se aqui precisa esse calculo, ou se pode mesmo só usar id
      ## Quase certeza que um teste que fiz dava errado usando só id em alguns casos, e esse era o bugfix

      f(d_array+id,par1,par2,x,y)
      #f(d_array+id,par1,par2,idX,idY)
    end
  end
  def map_2para_coord_2D(d_array, par1, par2, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_2para_coord_2D_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,par2,sizeX,sizeY,f])
    d_array
  end
## MAP = 1 GNX; 2 PARAMETERS; 2D, COORD: TRUE, RETURN: TRUE;
  defk map_2para_coord_2D_resp_kernel(d_array, ret, par1, par2, step, sizeX, sizeY, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    idY = threadIdx.y + blockIdx.y * blockDim.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      ret[id] = f(d_array[id], par1, par2, x, y)
      #ret[id] = f(d_array[id], par1, par2, idX, idY)
    end
  end
  def map_2para_coord_2D_resp(d_array, par1, par2, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_2para_coord_2D_resp_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,par2,step,sizeX,sizeY,f])
    ret
  end

## MAP = 1 GNX; 3 PARAMETERS; 1D;
  defk map_3para_1D_kernel(d_array, par1, par2, par3, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array[id],par1,par2,par3)
    end
  end
  def map_3para_1D(d_array, par1, par2, par3, f) do
      block_size =  128;
      {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
      #size = l*step
      size = l
      nBlocks = floor ((size + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map_3para_1D_kernel/7,{nBlocks,1,1},{block_size,1,1},[d_array,par1,par2,par3,step,size,f])
      d_array
  end
## MAP = 1 GNX; 3 PARAMETERS; 1D, RETURN: TRUE;
  defk map_3para_1D_resp_kernel(d_array, ret, par1, par2, par3, step, size, f) do
        idX = blockIdx.x * blockDim.x + threadIdx.x
        stride = blockDim.x * gridDim.x

        for i in range(idX,size,stride) do
          id = i*step
          ret[id] = f(d_array[id],par1,par2,par3)
        end
  end
  def map_3para_1D_resp(d_array, par1, par2, par3, f) do
      block_size =  128;
      {l,step} = case PolyHok.get_shape_gnx(d_array) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
      #size = l*step
      size = l
      nBlocks = floor ((size + block_size - 1) / block_size)
      ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

      PolyHok.spawn(&Ske.map_3para_1D_resp_kernel/8,{nBlocks,1,1},{block_size,1,1},[d_array,ret,par1,par2,par3,step,size,f])
      ret
  end
## MAP = 1 GNX; 3 PARAMETERS; 1D, COORD: TRUE;
  defk map_3para_coord_1D_kernel(d_array, par1, par2, par3, step, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    #offset = x * blockDim.x * gridDim.x
    #if (offset < size) do
    #  f(d_array+id,par1,par2,par3,x)
    for i in range(idX,size,stride) do
      id = i*step
      f(d_array[id],par1,par2,par3,idX)
    end
  end
  def map_3para_coord_1D(d_array, par1, par2, par3, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    #size = sizeX*step
    size = sizeX

    block_size = 16
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_3para_coord_1D_kernel/7,{grid_rows,1,1},{block_size,block_size,1},[d_array,par1,par2,par3,step,size,f])
    d_array
  end
## MAP = 1 GNX; 3 PARAMETERS; 1D, COORD: TRUE, RETURN: TRUE;
  defk map_3para_coord_1D_resp_kernel(d_array, ret, par1, par2, par3, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array[id],par1,par2,par3,id)
    end
  end
  def map_3para_coord_1D_resp(d_array, par1, par2, par3, f) do
    {sizeX,step} =  case PolyHok.get_shape_gnx(d_array) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    #size = sizeX*step
    size = sizeX

    #block_size = 16
    block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    #PolyHok.spawn(&Ske.map_3para_coord_1D_resp_kernel/8,{grid_rows,1,1},{block_size,block_size,1},[d_array,ret,par1,par2,par3,step,size,f])
    PolyHok.spawn(&Ske.map_3para_coord_1D_resp_kernel/8,{grid_rows,1,1},{block_size,1,1},[d_array,ret,par1,par2,par3,step,size,f])
    ret
  end
## MAP = 1 GNX; 3 PARAMETERS; 2D;
  defk map_3para_2D_kernel(d_array, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    id = stride*step
    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      f(d_array[id], par1, par2, par3)
    end
  end
  def map_3para_2D(d_array, par1, par2, par3, f) do
      #block_size =  128;
      {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      #nBlocks = floor ((size + block_size - 1) / block_size)
      block_size = 16
      grid_rows = trunc ((sizeX + block_size - 1) / block_size)
      grid_cols = trunc ((sizeY + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map_3para_2D_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,par1,par2,par3,step,sizeX,sizeY,f])
      d_array
  end
## MAP = 1 GNX; 3 PARAMETERS; 2D, RETURN: TRUE;
  defk map_3para_2D_resp_kernel(d_array, ret, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    id = stride*step
    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      ret[id] = f(d_array[id], par1, par2, par3)
    end
  end
  def map_3para_2D_resp(d_array, par1, par2, par3, f) do
      #block_size =  128;
      {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      #nBlocks = floor ((size + block_size - 1) / block_size)
      block_size = 16
      grid_rows = trunc ((sizeX + block_size - 1) / block_size)
      grid_cols = trunc ((sizeY + block_size - 1) / block_size)
      ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

      PolyHok.spawn(&Ske.map_3para_2D_resp_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,par2,par3,step,sizeX,sizeY,f])
      ret
  end
## MAP = 1 GNX; 3 PARAMETERS; 2D, COORD: TRUE;
  defk map_3para_coord_2D_kernel(d_array, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    idY = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x

    id  = step * offset
    ## Aqui tenho que verificar o 'step'
    if (offset < (sizeX*sizeY)) do
    #if (offset < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY

      f(d_array+id,par1,par2,par3,x,y)
    end
  end
  def map_3para_coord_2D(d_array, par1, par2, par3, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map_3para_coord_2D_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,par1,par2,par3,step,sizeX,sizeY,f])
    d_array
  end
## MAP = 1 GNX; 3 PARAMETERS; 2D, COORD: TRUE, RETURN: TRUE;
  defk map_3para_coord_2D_resp_kernel(d_array, ret, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      ret[id] = f(d_array[id], par1, par2, par3, x, y)
    end
  end
  def map_3para_coord_2D_resp(d_array, par1, par2, par3, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array),PolyHok.get_type(d_array))

    PolyHok.spawn(&Ske.map_3para_coord_2D_resp_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,ret,par1,par2,par3,step,sizeX,sizeY,f])
    ret
  end

## 2 GNX
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## "X" indica os que adicionei depois, e portanto preciso testar pra confirmar que fiz corretamente
## "Y" indica o que eu fiz modificação para corrigir, e precisa ver se é correto substituir em todos outros casos
## ""Z é a mesma coisa que o "Y", porém modifiquei mais radicalmente (era um Y, que já precisava revisar, e agora modifiquei totalmente, então precisa atenção dubplicada). Essa modificação em especifico diminuiu muito o tempo de execução, ficando mais parecida com a implementação no dot product com poly_hok puro

## X MAP = 2 GNX; 0 PARAMETERS; 1D;
  defk map2_0para_1D_kernel(d_array1, d_array2, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id])
    end
  end
  def map2_0para_1D(d_array1, d_array2, f) do
    block_size =  256;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    if(l1 != l2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end
    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_0para_1D_kernel/5,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,step1,size,f])
    d_array1
  end
## Z MAP = 2 GNX; 0 PARAMETERS; 1D, RETURN: TRUE;
  defk map2_0para_1D_resp_kernel(d_array1, d_array2, ret, step, size, f) do
    # id = blockIdx.x * blockDim.x + threadIdx.x
    # stride = blockDim.x * gridDim.x
    # size2 = size*step

    # # printf("outside id thread: %d\\n",id);
    # # printf("size: %d\\n",size);
    # for i in range(id, size2, stride) do
    #   # printf("inside id thread: %d\\n",id);
    #   # id2 = i*step
    #   id2 = i
    # # printf("posicao no array: %d\\n",id);
    #   ret[id2] = f(d_array1[id2],d_array2[id2])
    # end

    id = blockIdx.x * blockDim.x + threadIdx.x
    if(id < size) do
      ret[id] = f(d_array1[id],d_array2[id])
    end
  end
  def map2_0para_1D_resp(d_array1, d_array2, f) do
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
      {l} -> {l,1}
      {l,step} -> {l,step}
      x -> raise "Invalid shape for 1D map: #{inspect x}!"
    end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
      {l} -> {l,1}
      {l,step} -> {l,step}
      x -> raise "Invalid shape for 1D map: #{inspect x}!"
    end
    if(l1 != l2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    block_size =  256;
    size = l1*step1
    # size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape_gnx(d_array1),PolyHok.get_type_gnx(d_array1))

    PolyHok.spawn(&Ske.map2_0para_1D_resp_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,ret,step1,size,f])
    # PolyHok.spawn(&Ske.map2_0para_1D_resp_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,ret,size,step1,f])
    ret
  end
## X MAP = 2 GNX; 0 PARAMETERS; 1D, COORD: TRUE;
  defk map2_0para_coord_1D_kernel(d_array1, d_array2, step, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id],idX)
    end
  end
  def map2_0para_coord_1D(d_array1, d_array2, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_0para_coord_1D_kernel/5,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,step1,size,f])
    d_array1
  end
## X MAP = 2 GNX; 0 PARAMETERS; 1D, COORD: TRUE, RETURN: TRUE;
  defk map2_0para_coord_1D_resp_kernel(d_array1, d_array2, ret, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],idX)
    end
  end
  def map2_0para_coord_1D_resp(d_array1, d_array2, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_0para_coord_1D_resp_kernel/6,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,ret,step1,size,f])
    ret
  end
## X MAP = 2 GNX; 0 PARAMETERS; 2D;
  defk map2_0para_2D_kernel(d_array1, d_array2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      id = stride*step
      f(d_array1[id],d_array2[id])
    end
  end
  def map2_0para_2D(d_array1, d_array2, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_0para_2D_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,step1,sizeX1,sizeY1,f])
    d_array1
  end
## X MAP = 2 GNX; 0 PARAMETERS; 2D, RETURN: TRUE;
  defk map2_0para_2D_resp_kernel(d_array1, d_array2, ret, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      ret[id] = f(d_array1[id],d_array2[id])
    end
  end
  def map2_0para_2D_resp(d_array1, d_array2, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_0para_2D_resp_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,ret,step1,sizeX1,sizeY1,f])
    ret
  end
## X MAP = 2 GNX; 0 PARAMETERS; 2D, COORD: TRUE;
  defk map2_0para_coord_2D_kernel(d_array1, d_array2, step, sizeX, sizeY, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    idY = threadIdx.y + blockIdx.y * blockDim.y
    stride = idX + idY * blockDim.x * gridDim.x

    id  = step * stride
    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY

      f(d_array1+id,d_array2+id,x,y)
    end
  end
  def map2_0para_coord_2D(d_array1, d_array2, f) do
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_0para_coord_2D_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,step1,sizeX1,sizeY1,f])
    #PolyHok.spawn(&Ske.map2_0para_coord_2D_kernel/6,{sizeX,sizeX,1},{1,1,1},[d_array1,d_array2,step,par1,sizeX,sizeY,f])
    d_array1
  end
## X MAP = 2 GNX; 0 PARAMETERS; 2D, COORD: TRUE, RETURN: TRUE;
  defk map2_0para_coord_2D_resp_kernel(d_array1, d_array2, ret, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id  = step * stride

      ret[id] = f(d_array1[id], d_array2[id], x, y)
    end
  end
  def map2_0para_coord_2D_resp(d_array1, d_array2, f) do
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_0para_coord_2D_resp_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,ret,step1,sizeX1,sizeY1,f])
    ret
  end

## X MAP = 2 GNX; 1 PARAMETER; 1D;
  defk map2_1para_1D_kernel(d_array1, d_array2, par1, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      f(d_array1[id], d_array2[id], par1)
    end
  end
  def map2_1para_1D(d_array1, d_array2, par1, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    if(l1 != l2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end
    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_1para_1D_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,par1,step1,size,f])
    d_array1
  end
## X MAP = 2 GNX; 1 PARAMETER; 1D, RETURN: TRUE;
  defk map2_1para_1D_resp_kernel(d_array1, d_array2, ret, par1, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id], d_array2[id], par1)
    end
  end
  def map2_1para_1D_resp(d_array1, d_array2, par1, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    if(l1 != l2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_1para_1D_resp_kernel/7,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,ret,par1,step1,size,f])
    ret
  end
## X MAP = 2 GNX; 1 PARAMETER; 1D, COORD: TRUE;
  defk map2_1para_coord_1D_kernel(d_array1, d_array2, par1, step, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id],par1,idX)
    end
  end
  def map2_1para_coord_1D(d_array1, d_array2, par1, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end
    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_1para_coord_1D_kernel/6,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,par1,step1,size,f])
    d_array1
  end
## X MAP = 2 GNX; 1 PARAMETER; 1D, COORD: TRUE, RETURN: TRUE;
  defk map2_1para_coord_1D_resp_kernel(d_array1, d_array2, ret, par1, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],par1,idX)
    end
  end
  def map2_1para_coord_1D_resp(d_array1, d_array2, par1, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_1para_coord_1D_resp_kernel/7,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,ret,par1,step1,size,f])
    ret
  end
## X MAP = 2 GNX; 1 PARAMETER; 2D;
  defk map2_1para_2D_kernel(d_array1, d_array2, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      id = stride*step
      f(d_array1[id], d_array2[id], par1)
    end
  end
  def map2_1para_2D(d_array1, d_array2, par1, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_1para_2D_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,par1,step1,sizeX1,sizeY1,f])
    d_array1
  end
## X MAP = 2 GNX; 1 PARAMETER; 2D, RETURN:TRUE;
  defk map2_1para_2D_resp_kernel(d_array1, d_array2, ret, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      ret[id] = f(d_array1[id], d_array2[id], par1)
    end
  end
  def map2_1para_2D_resp(d_array1, d_array2, par1, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_1para_2D_resp_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,ret,par1,step1,sizeX1,sizeY1,f])
    ret
  end
## MAP = 2 GNX; 1 PARAMETER; 2D, COORD: TRUE;
  defk map2_1para_coord_2D_kernel(d_array1, d_array2, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      f(d_array1+id, d_array2+id, par1, x, y)
    end
  end
  def map2_1para_coord_2D(d_array1, d_array2, par1, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeY != sizeY2 or step != step2) do
      raise "Both matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_1para_coord_2D_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,par1,step,sizeX,sizeY,f])
    d_array1
  end
## MAP = 2 GNX; 1 PARAMETER; 2D, RETURN: TRUE;
  defk map2_1para_coord_2D_resp_kernel(d_array1, d_array2, ret, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      ret[id] = f(d_array1[id], d_array2[id], par1, x, y)
    end
  end
  def map2_1para_coord_2D_resp(d_array1, d_array2, par1, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeY != sizeY2 or step != step2) do
      raise "Both matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_1para_coord_2D_resp_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,ret,par1,step,sizeX,sizeY,f])
    ret
  end

## MAP = 2 GNX; 2 PARAMETERS; 1D;
  defk map2_2para_1D_kernel(d_array1, d_array2, par1, par2, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(idX < size*step) do
      #id = stride*step
      id = idX*step

      #printf("%d %d ",idX,id)
      f(d_array1+idX, d_array2+id, par1, par2)  # Porque um é idX e outro é id?
    end
  end
  def map2_2para_1D(d_array1, d_array2, par1, par2, f) do
    block_size =  128;
    # {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
    {l1,_step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    # {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
    {_l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    # if(l1 != l2 or step1 != step2) do
    #   IO.inspect({l1,step1})
    #   IO.inspect({l2,step2})
    #   raise "Both matrices shall have same shape."
    # end

    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_2para_1D_kernel/7,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,par1,par2,step2,size,f])
    d_array1
  end
## X MAP = 2 GNX; 2 PARAMETERS; 1D, RETURN: TRUE;
  defk map2_2para_1D_resp_kernel(d_array1, d_array2, ret, par1, par2, step, size, f) do
    #printf("AQUI\\n")
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],par1,par2)
    end
  end
  def map2_2para_1D_resp(d_array1, d_array2, par1, par2, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    if(l1 != l2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_2para_1D_resp_kernel/8,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,ret,par1,par2,step1,size,f])
    ret
  end
## X MAP = 2 GNX; 2 PARAMETERS; 1D, COORD: TRUE;
  defk map2_2para_coord_1D_kernel(d_array1, d_array2, par1, par2, step, size, f) do
    x = threadIdx.x + blockIdx.x * blockDim.x
    offset = x * blockDim.x * gridDim.x

    if (offset < size) do
      id = step * offset
      f(d_array1[id],d_array2[id],par1,par2,x)
    end
  end
  def map2_2para_coord_1D(d_array1, d_array2, par1, par2, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    if(sizeX1 != sizeX2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end
    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_2para_coord_1D_kernel/7,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,par1,par2,step1,size,f])
    d_array1
  end
## X MAP = 2 GNX; 2 PARAMETERS; 1D, COORD: TRUE, RETURN: TRUE;
  defk map2_2para_coord_1D_resp_kernel(d_array1, d_array2, ret, par1, par2, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    #offset = idX * blockDim.x * gridDim.x
    #if(offset < size)do
    #  ret[offset] = f(d_array[offset],par1,par2,idX)
    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],par1,par2,idX)
    end
  end
  def map2_2para_coord_1D_resp(d_array1,d_array2, par1, par2, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    if(sizeX1 != sizeX2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_2para_coord_1D_resp_kernel/8,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,ret,par1,par2,step1,size,f])
    ret
  end
## X MAP = 2 GNX; 2 PARAMETERS; 2D;
  defk map2_2para_2D_kernel(d_array1, d_array2, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      f(d_array1[id], d_array2[id], par1, par2)
    end
  end
  def map2_2para_2D(d_array1, d_array2, par1, par2, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_2para_2D_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,par1,par2,step1,sizeX1,sizeY1,f])
    d_array1
  end
## X MAP = 2 GNX; 2 PARAMETERS; 2D, RETURN: TRUE;
  defk map2_2para_2D_resp_kernel(d_array1, d_array2, ret, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      ret[id] = f(d_array1[id], d_array2[id], par1, par2)
    end
  end
  def map2_2para_2D_resp(d_array1, d_array2, par1, par2, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_2para_2D_resp_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,ret,par1,par2,step1,sizeX1,sizeY1,f])
    ret
  end
## MAP = 2 GNX; 2 PARAMETERS; 2D, COORD: TRUE;
  defk map2_2para_coord_2D_kernel(d_array1, d_array2, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      f(d_array1+id, d_array2+id, par1, par2, x, y)
    end
  end
  def map2_2para_coord_2D(d_array1, d_array2, par1, par2, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeY != sizeY2 or step != step2) do
      raise "Both matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_2para_coord_2D_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,par1,par2,step,sizeX,sizeY,f])
    d_array1
  end
## MAP = 2 GNX; 2 PARAMETERS; 2D, COORD: TRUE, RETURN: TRUE;
  defk map2_2para_coord_2D_resp_kernel(d_array1, d_array2, ret, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      ret[id] = f(d_array1[id], d_array2[id], par1, par2, x, y)
    end
  end
  def map2_2para_coord_2D_resp(d_array1, d_array2, par1, par2, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeY != sizeY2 or step != step2) do
      raise "Both matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_2para_coord_2D_resp_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,ret,par1,par2,step,sizeX,sizeY,f])
    ret
  end

## X MAP = 2 GNX; 3 PARAMETERS; 1D;
  defk map2_3para_1D_kernel(d_array1, d_array2, par1, par2, par3, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id],par1,par2,par3)
    end
  end
  def map2_3para_1D(d_array1, d_array2, par1, par2, par3, f) do
      block_size =  128;
      {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
      {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
      if(l1 != l2 or step1 != step2) do
        raise "Both matrices shall have same shape."
      end

      #size = l*step
      size = l1
      nBlocks = floor ((size + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map2_3para_1D_kernel/8,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,par1,par2,par3,step1,size,f])
      d_array1
  end
## X MAP = 2 GNX; 3 PARAMETERS; 1D, RETURN: TRUE;
  defk map2_3para_1D_resp_kernel(d_array1, d_array2, ret, par1, par2, par3, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],par1,par2,par3)
    end
  end
  def map2_3para_1D_resp(d_array1, d_array2, par1, par2, par3, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
              {l} -> {l,1}
              {l,step} -> {l,step}
              x -> raise "Invalid shape for 1D map: #{inspect x}!"
            end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
              {l} -> {l,1}
              {l,step} -> {l,step}
              x -> raise "Invalid shape for 1D map: #{inspect x}!"
            end
    if(l1 != l2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_3para_1D_resp_kernel/9,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,ret,par1,par2,par3,step1,size,f])
    ret
  end
## X MAP = 2 GNX; 3 PARAMETERS; 1D, COORD: TRUE;
  defk map2_3para_coord_1D_kernel(d_array1, d_array2, par1, par2, par3, step, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    #offset = x * blockDim.x * gridDim.x
    #if (offset < size) do
    #  f(d_array+id,par1,par2,par3,x)
    for i in range(idX,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id],par1,par2,par3,idX)
    end
  end
  def map2_3para_coord_1D(d_array1, d_array2, par1, par2, par3, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    if(sizeX1 != sizeX2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_3para_coord_1D_kernel/8,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,par1,par2,par3,step1,size,f])
    d_array1
  end
## X MAP = 2 GNX; 3 PARAMETERS; 1D, COORD: TRUE, RETURN: TRUE;
  defk map2_3para_coord_1D_resp_kernel(d_array1, d_array2, ret, par1, par2, par3, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],par1,par2,par3,id)
    end
  end
  def map2_3para_coord_1D_resp(d_array1, d_array2, par1, par2, par3, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    if(sizeX1 != sizeX2 or step1 != step2) do
      raise "Both matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    #block_size = 16
    block_size = 128
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    #PolyHok.spawn(&Ske.map_3para_coord_1D_resp_kernel/8,{grid_rows,1,1},{block_size,block_size,1},[d_array,ret,par1,par2,par3,step,size,f])
    PolyHok.spawn(&Ske.map2_3para_coord_1D_resp_kernel/9,{grid_rows,1,1},{block_size,1,1},[d_array1,d_array2,ret,par1,par2,par3,step1,size,f])
    ret
  end
## X MAP = 2 GNX; 3 PARAMETERS; 2D;
  defk map2_3para_2D_kernel(d_array1, d_array2, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    id = stride*step
    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      f(d_array2[id], d_array2[id], par1, par2, par3)
    end
  end
  def map2_3para_2D(d_array1, d_array2, par1, par2, par3, f) do
      #block_size =  128;
      {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
        raise "Both matrices shall have same shape."
      end

      #nBlocks = floor ((size + block_size - 1) / block_size)
      block_size = 16
      grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
      grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map2_3para_2D_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,par1,par2,par3,step1,sizeX1,sizeY1,f])
      d_array1
  end
## X MAP = 2 GNX; 3 PARAMETERS; 2D, RETURN: TRUE;
  defk map2_3para_2D_resp_kernel(d_array1, d_array2, ret, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    id = stride*step
    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      ret[id] = f(d_array1[id], d_array2[id], par1, par2, par3)
    end
  end
  def map2_3para_2D_resp(d_array1, d_array2, par1, par2, par3, f) do
      #block_size =  128;
      {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      if(sizeX1 != sizeX2 or sizeY1 != sizeY2 or step1 != step2) do
        raise "Both matrices shall have same shape."
      end

      #nBlocks = floor ((size + block_size - 1) / block_size)
      block_size = 16
      grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
      grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
      ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

      PolyHok.spawn(&Ske.map2_3para_2D_resp_kernel/10,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,ret,par1,par2,par3,step1,sizeX1,sizeY1,f])
      ret
  end
## MAP = 2 GNX; 3 PARAMETERS; 2D, COORD: TRUE;
  defk map2_3para_coord_2D_kernel(d_array1, d_array2, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      f(d_array1+id, d_array2+id, par1, par2, par3, x, y)
    end
  end
  def map2_3para_coord_2D(d_array1, d_array2, par1, par2, par3, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeY != sizeY2 or step != step2) do
      raise "Both matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map2_3para_coord_2D_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,par1,par2,par3,step,sizeX,sizeY,f])
    d_array1
  end
## MAP = 2 GNX; 3 PARAMETERS; 2D, COORD: TRUE, RETURN: TRUE;
  defk map2_3para_coord_2D_resp_kernel(d_array1, d_array2, ret, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      ret[id] = f(d_array1[id], d_array2[id], par1, par2, par3, x, y)
    end
  end
  def map2_3para_coord_2D_resp(d_array1, d_array2, par1, par2, par3, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeY != sizeY2 or step != step2) do
      raise "Both matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map2_3para_coord_2D_resp_kernel/10,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,ret,par1,par2,par3,step,sizeX,sizeY,f])
    ret
  end

## 3 GNX
## -----------------------------------------------------------------------------------------------------------------------------------------------------

## X MAP = 3 GNX; 0 PARAMETERS; 1D;
  defk map3_0para_1D_kernel(d_array1, d_array2, d_array3, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id],d_array3[id])
    end
  end
  def map3_0para_1D(d_array1, d_array2, d_array3, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l3,step3} = case PolyHok.get_shape_gnx(d_array3) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    if(l1 != l2 or l2 != l3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end
    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_0para_1D_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,d_array3,step1,size,f])
    d_array1
  end
## X MAP = 3 GNX; 0 PARAMETERS; 1D, RETURN: TRUE;
  defk map3_0para_1D_resp_kernel(d_array1, d_array2, d_array3, ret, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],d_array3[id])
    end
  end
  def map3_0para_1D_resp(d_array1, d_array2, d_array3, f) do
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
    end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
    end
    {l3,step3} = case PolyHok.get_shape_gnx(d_array3) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
    end
    if(l1 != l2 or l2 != l3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    block_size =  128;
    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_0para_1D_resp_kernel/7,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,d_array3,ret,step1,size,f])
    ret
  end
## X MAP = 3 GNX; 0 PARAMETERS; 1D, COORD: TRUE;
  defk map3_0para_coord_1D_kernel(d_array1, d_array2, d_array3, step, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id],d_array3[id],idX)
    end
  end
  def map3_0para_coord_1D(d_array1, d_array2, d_array3, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_0para_coord_1D_kernel/6,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,step1,size,f])
    d_array1
  end
## X MAP = 3 GNX; 0 PARAMETERS; 1D, COORD: TRUE, RETURN: TRUE;
  defk map3_0para_coord_1D_resp_kernel(d_array1, d_array2, d_array3, ret, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],d_array3[id],idX)
    end
  end
  def map3_0para_coord_1D_resp(d_array1, d_array2, d_array3, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_0para_coord_1D_resp_kernel/7,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,step1,size,f])
    ret
  end
## X MAP = 3 GNX; 0 PARAMETERS; 2D;
  defk map3_0para_2D_kernel(d_array1, d_array2, d_array3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      id = stride*step
      f(d_array1[id],d_array2[id],d_array3[id])
    end
  end
  def map3_0para_2D(d_array1, d_array2, d_array3, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_0para_2D_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,step1,sizeX1,sizeY1,f])
    d_array1
  end
## X MAP = 3 GNX; 0 PARAMETERS; 2D, RETURN: TRUE;
  defk map3_0para_2D_resp_kernel(d_array1, d_array2, d_array3, ret, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      ret[id] = f(d_array1[id],d_array2[id],d_array3[id])
    end
  end
  def map3_0para_2D_resp(d_array1, d_array2, d_array3, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_0para_2D_resp_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,step1,sizeX1,sizeY1,f])
    ret
  end
## X MAP = 3 GNX; 0 PARAMETERS; 2D, COORD: TRUE;
  defk map3_0para_coord_2D_kernel(d_array1, d_array2, d_array3, step, sizeX, sizeY, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    idY = threadIdx.y + blockIdx.y * blockDim.y
    stride = idX + idY * blockDim.x * gridDim.x

    id  = step * stride
    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY

      f(d_array1+id,d_array2+id,d_array3+id,x,y)
    end
  end
  def map3_0para_coord_2D(d_array1, d_array2, d_array3, f) do
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_0para_coord_2D_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,step1,sizeX1,sizeY1,f])
    #PolyHok.spawn(&Ske.map3_0para_coord_2D_kernel/7,{sizeX,sizeX,1},{1,1,1},[d_array1,d_array2,d_array3,step,par1,sizeX,sizeY,f])
    d_array1
  end
## X MAP = 3 GNX; 0 PARAMETERS; 2D, COORD: TRUE, RETURN: TRUE;
  defk map3_0para_coord_2D_resp_kernel(d_array1, d_array2, d_array3, ret, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id  = step * stride

      ret[id] = f(d_array1[id], d_array2[id], d_array3[id], x, y)
    end
  end
  def map3_0para_coord_2D_resp(d_array1, d_array2, d_array3, f) do
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_0para_coord_2D_resp_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,step1,sizeX1,sizeY1,f])
    ret
  end

## X MAP = 3 GNX; 1 PARAMETER; 1D;
  defk map3_1para_1D_kernel(d_array1, d_array2, d_array3, par1, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      f(d_array1[id], d_array2[id], d_array3[id], par1)
    end
  end
  def map3_1para_1D(d_array1, d_array2, d_array3, par1, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l3,step3} = case PolyHok.get_shape_gnx(d_array3) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    if(l1 != l2  or l2 != l3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end
    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_1para_1D_kernel/7,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,d_array3,par1,step1,size,f])
    d_array1
  end
## X MAP = 3 GNX; 1 PARAMETER; 1D, RETURN: TRUE;
  defk map3_1para_1D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, step, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id], d_array2[id], d_array3[id], par1)
    end
  end
  def map3_1para_1D_resp(d_array1, d_array2, d_array3, par1, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l3,step3} = case PolyHok.get_shape_gnx(d_array3) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    if(l1 != l2 or l2 != l3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_1para_1D_resp_kernel/8,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,d_array3,ret,par1,step1,size,f])
    ret
  end
## X MAP = 3 GNX; 1 PARAMETER; 1D, COORD: TRUE;
  defk map3_1para_coord_1D_kernel(d_array1, d_array2, d_array3, par1, step, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id],d_array3[id],par1,idX)
    end
  end
  def map3_1para_coord_1D(d_array1, d_array2, d_array3, par1, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end
    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_1para_coord_1D_kernel/7,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,par1,step1,size,f])
    d_array1
  end
## X MAP = 3 GNX; 1 PARAMETER; 1D, COORD: TRUE, RETURN: TRUE;
  defk map3_1para_coord_1D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],d_array3[id],par1,idX)
    end
  end
  def map3_1para_coord_1D_resp(d_array1, d_array2, d_array3, par1, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    {sizeX3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l} -> {l,1}
                            {l,step} -> {l,step}
                            x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_1para_coord_1D_resp_kernel/8,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,par1,step1,size,f])
    ret
  end
## X MAP = 3 GNX; 1 PARAMETER; 2D;
  defk map3_1para_2D_kernel(d_array1, d_array2, d_array3, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if (stride < (sizeX*sizeY)) do
    #if (stride < (sizeX*sizeY*step)) do
      id = stride*step
      f(d_array1[id], d_array2[id], d_array3[id], par1)
    end
  end
  def map3_1para_2D(d_array1, d_array2, d_array3, par1, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_1para_2D_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,par1,step1,sizeX1,sizeY1,f])
    d_array1
  end
## X MAP = 3 GNX; 1 PARAMETER; 2D, RETURN: TRUE;
  defk map3_1para_2D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      ret[id] = f(d_array1[id], d_array2[id], d_array3[id], par1)
    end
  end
  def map3_1para_2D_resp(d_array1, d_array2, d_array3, par1, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_1para_2D_resp_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,par1,step1,sizeX1,sizeY1,f])
    ret
  end
# \/ ESSE É O QUE MUDEI o INDEX pra funcinar com MM, onde ele tava acessando posição ilegal da memória pela forma de calcular a posição (usa calculo da posição 0 do array, mas eu tava passando posição id)
## MAP = 3 GNX; 1 PARAMETER; 2D, COORD: TRUE;
  defk map3_1para_coord_2D_kernel(d_array1, d_array2, d_array3, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    # if(stride < (sizeX*sizeY*step)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step
      # id = stride

      #f(d_array1+id, d_array2+id, d_array3+id, par1, x, y)
      f(d_array1, d_array2, d_array3, par1, x, y)
    end
  end
  def map3_1para_coord_2D(d_array1, d_array2, d_array3, par1, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeX2 != sizeX3 or sizeY != sizeY2 or sizeY2 != sizeY3 or step != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    # grid_rows = trunc (((sizeX*step) + block_size - 1) / block_size)
    # grid_cols = trunc (((sizeY*step) + block_size - 1) / block_size)

    # IO.inspect(d_array2)
    PolyHok.spawn(&Ske.map3_1para_coord_2D_kernel/8,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,par1,step,sizeX,sizeY,f])
    # IO.inspect(PolyHok.get_gnx(d_array2))
    # IO.inspect(d_array1)
    d_array1
  end
## X MAP = 3 GNX; 1 PARAMETER; 2D, COORD: TRUE, RETURN: TRUE;
  defk map3_1para_coord_2D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      ret[id] = f(d_array1[id], d_array2[id], d_array3[id], par1, x, y)
    end
  end
  def map3_1para_coord_2D_resp(d_array1, d_array2, d_array3, par1, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeX2 != sizeX3 or sizeY != sizeY2 or sizeY2 != sizeY3 or step != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_1para_coord_2D_resp_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,par1,step,sizeX,sizeY,f])
    ret
  end

## X MAP = 3 GNX; 2 PARAMETERS; 1D;
  defk map3_2para_1D_kernel(d_array1, d_array2, d_array3, par1, par2, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(idX < size*step) do
      #id = stride*step
      id = idX*step

      #printf("%d %d ",idX,id)
      f(d_array1+idX, d_array2+id, d_array3+id, par1, par2)
    end
  end
  def map3_2para_1D(d_array1, d_array2, d_array3, par1, par2, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l3,step3} = case PolyHok.get_shape_gnx(d_array3) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    if(l1 != l2 or l2 != l3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_2para_1D_kernel/8,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,d_array3,par1,par2,step1,size,f])
    d_array1
  end
## X MAP = 3 GNX; 2 PARAMETERS; 1D, RETURN: TRUE;
  defk map3_2para_1D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, par2, step, size, f) do
    #printf("AQUI\\n")
    id = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(id,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],d_array3[id],par1,par2)
    end
  end
  def map3_2para_1D_resp(d_array1, d_array2, d_array3, par1, par2, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    {l3,step3} = case PolyHok.get_shape_gnx(d_array3) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
    if(l1 != l2 or l2 != l3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_2para_1D_resp_kernel/9,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,d_array3,ret,par1,par2,step1,size,f])
    ret
  end
## X MAP = 3 GNX; 2 PARAMETERS; 1D, COORD: TRUE;
  defk map3_2para_coord_1D_kernel(d_array1, d_array2, d_array3, par1, par2, step, size, f) do
    x = threadIdx.x + blockIdx.x * blockDim.x
    offset = x * blockDim.x * gridDim.x

    if (offset < size) do
      id = step * offset
      f(d_array1[id],d_array2[id],d_array3[id],par1,par2,x)
    end
  end
  def map3_2para_coord_1D(d_array1, d_array2, d_array3, par1, par2, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end
    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_2para_coord_1D_kernel/8,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,par1,par2,step1,size,f])
    d_array1
  end
## X MAP = 3 GNX; 2 PARAMETERS; 1D, COORD: TRUE, RETURN: TRUE;
  defk map3_2para_coord_1D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, par2, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    #offset = idX * blockDim.x * gridDim.x
    #if(offset < size)do
    #  ret[offset] = f(d_array[offset],par1,par2,idX)
    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],d_array3[id],par1,par2,idX)
    end
  end
  def map3_2para_coord_1D_resp(d_array1,d_array2, d_array3, par1, par2, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_2para_coord_1D_resp_kernel/9,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,par1,par2,step1,size,f])
    ret
  end
## X MAP = 3 GNX; 2 PARAMETERS; 2D;
  defk map3_2para_2D_kernel(d_array1, d_array2, d_array3, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      f(d_array1[id], d_array2[id], d_array3[id], par1, par2)
    end
  end
  def map3_2para_2D(d_array1, d_array2, d_array3, par1, par2, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_2para_2D_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,par1,par2,step1,sizeX1,sizeY1,f])
    d_array1
  end
## X MAP = 3 GNX; 2 PARAMETERS; 2D, RETURN: TRUE;
  defk map3_2para_2D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      id = stride*step
      ret[id] = f(d_array1[id], d_array2[id], d_array3[id], par1, par2)
    end
  end
  def map3_2para_2D_resp(d_array1, d_array2, d_array3, par1, par2, f) do
    #block_size =  128;
    {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #nBlocks = floor ((size + block_size - 1) / block_size)
    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_2para_2D_resp_kernel/10,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,par1,par2,step1,sizeX1,sizeY1,f])
    ret
  end
## X MAP = 3 GNX; 2 PARAMETERS; 2D, COORD: TRUE;
  defk map3_2para_coord_2D_kernel(d_array1, d_array2, d_array3, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      f(d_array1+id, d_array2+id, d_arraye3+id, par1, par2, x, y)
    end
  end
  def map3_2para_coord_2D(d_array1, d_array2, d_array3, par1, par2, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeX2 != sizeX3 or sizeY != sizeY2 or sizeY2 != sizeY3 or step != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_2para_coord_2D_kernel/9,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,par1,par2,step,sizeX,sizeY,f])
    d_array1
  end
## X MAP = 3 GNX; 2 PARAMETERS; 2D, COORD: TRUE, RETURN: TRUE;
  defk map3_2para_coord_2D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, par2, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      ret[id] = f(d_array1[id], d_array2[id], d_array3[id], par1, par2, x, y)
    end
  end
  def map3_2para_coord_2D_resp(d_array1, d_array2, d_array3, par1, par2, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeX3 != sizeX3 or sizeY != sizeY2 or sizeY2 != sizeY3 or step != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_2para_coord_2D_resp_kernel/10,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,par1,par2,step,sizeX,sizeY,f])
    ret
  end

## X MAP = 3 GNX; 3 PARAMETERS; 1D;
  defk map3_3para_1D_kernel(d_array1, d_array2, d_array3, par1, par2, par3, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id],d_array3[id],par1,par2,par3)
    end
  end
  def map3_3para_1D(d_array1, d_array2, d_array3, par1, par2, par3, f) do
      block_size =  128;
      {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
      {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
      {l3,step3} = case PolyHok.get_shape_gnx(d_array3) do
                {l} -> {l,1}
                {l,step} -> {l,step}
                x -> raise "Invalid shape for 1D map: #{inspect x}!"
              end
      if(l1 != l2 or l2 != l3 or step1 != step2 or step2 != step3) do
        raise "All matrices shall have same shape."
      end

      #size = l*step
      size = l1
      nBlocks = floor ((size + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map3_3para_1D_kernel/9,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,d_array3,par1,par2,par3,step1,size,f])
      d_array1
  end
## X MAP = 3 GNX; 3 PARAMETERS; 1D, RETURN: TRUE;
  defk map3_3para_1D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, par2, par3, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],d_array3[id],par1,par2,par3)
    end
  end
  def map3_3para_1D_resp(d_array1, d_array2, d_array3, par1, par2, par3, f) do
    block_size =  128;
    {l1,step1} = case PolyHok.get_shape_gnx(d_array1) do
              {l} -> {l,1}
              {l,step} -> {l,step}
              x -> raise "Invalid shape for 1D map: #{inspect x}!"
            end
    {l2,step2} = case PolyHok.get_shape_gnx(d_array2) do
              {l} -> {l,1}
              {l,step} -> {l,step}
              x -> raise "Invalid shape for 1D map: #{inspect x}!"
            end
    {l3,step3} = case PolyHok.get_shape_gnx(d_array3) do
              {l} -> {l,1}
              {l,step} -> {l,step}
              x -> raise "Invalid shape for 1D map: #{inspect x}!"
            end
    if(l1 != l2 or l2 != l3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #size = l*step
    size = l1
    nBlocks = floor ((size + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_3para_1D_resp_kernel/10,{nBlocks,1,1},{block_size,1,1},[d_array1,d_array2,d_array3,ret,par1,par2,par3,step1,size,f])
    ret
  end
## X MAP = 3 GNX; 3 PARAMETERS; 1D, COORD: TRUE;
  defk map3_3para_coord_1D_kernel(d_array1, d_array2, d_array3, par1, par2, par3, step, size, f) do
    idX = threadIdx.x + blockIdx.x * blockDim.x
    stride = blockDim.x * gridDim.x

    #offset = x * blockDim.x * gridDim.x
    #if (offset < size) do
    #  f(d_array+id,par1,par2,par3,x)
    for i in range(idX,size,stride) do
      id = i*step
      f(d_array1[id],d_array2[id],d_array3[id],par1,par2,par3,idX)
    end
  end
  def map3_3para_coord_1D(d_array1, d_array2, d_array3, par1, par2, par3, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    block_size = 16
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_3para_coord_1D_kernel/9,{grid_rows,1,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,par1,par2,par3,step1,size,f])
    d_array1
  end
## X MAP = 3 GNX; 3 PARAMETERS; 1D, COORD: TRUE, RETURN: TRUE;
  defk map3_3para_coord_1D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, par2, par3, step, size, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(idX,size,stride) do
      id = i*step
      ret[id] = f(d_array1[id],d_array2[id],par1,par2,par3,id)
    end
  end
  def map3_3para_coord_1D_resp(d_array1, d_array2, d_array3, par1, par2, par3, f) do
    {sizeX1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    {sizeX3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                          {l} -> {l,1}
                          {l,step} -> {l,step}
                          x -> raise "Invalid shape for a 1D map: #{inspect x}!"
                        end
    if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or step1 != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    #size = sizeX*step
    size = sizeX1

    #block_size = 16
    block_size = 128
    grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    #PolyHok.spawn(&Ske.map3_3para_coord_1D_resp_kernel/10,{grid_rows,1,1},{block_size,block_size,1},[d_array,d_array2,d_array3,ret,par1,par2,par3,step,size,f])
    PolyHok.spawn(&Ske.map3_3para_coord_1D_resp_kernel/10,{grid_rows,1,1},{block_size,1,1},[d_array1,d_array2,d_array3,ret,par1,par2,par3,step1,size,f])
    ret
  end
## X MAP = 3 GNX; 3 PARAMETERS; 2D;
  defk map3_3para_2D_kernel(d_array1, d_array2, d_array3, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    id = stride*step
    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      f(d_array2[id], d_array2[id], d_array3[id], par1, par2, par3)
    end
  end
  def map3_3para_2D(d_array1, d_array2, d_array3, par1, par2, par3, f) do
      #block_size =  128;
      {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
        raise "All matrices shall have same shape."
      end

      #nBlocks = floor ((size + block_size - 1) / block_size)
      block_size = 16
      grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
      grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map3_3para_2D_kernel/10,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,par1,par2,par3,step1,sizeX1,sizeY1,f])
      d_array1
  end
## X MAP = 3 GNX; 3 PARAMETERS; 2D, RETURN: TRUE;
  defk map3_3para_2D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    id = stride*step
    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
    #if(stride < (sizeX*sizeY*step)) do
      ret[id] = f(d_array1[id], d_array2[id], d_array3[id], par1, par2, par3)
    end
  end
  def map3_3para_2D_resp(d_array1, d_array2, d_array3, par1, par2, par3, f) do
      #block_size =  128;
      {sizeX1,sizeY1,step1} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
      if(sizeX1 != sizeX2 or sizeX2 != sizeX3 or sizeY1 != sizeY2 or sizeY2 != sizeY3 or step1 != step2 or step2 != step3) do
        raise "All matrices shall have same shape."
      end

      #nBlocks = floor ((size + block_size - 1) / block_size)
      block_size = 16
      grid_rows = trunc ((sizeX1 + block_size - 1) / block_size)
      grid_cols = trunc ((sizeY1 + block_size - 1) / block_size)
      ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

      PolyHok.spawn(&Ske.map3_3para_2D_resp_kernel/11,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,par1,par2,par3,step1,sizeX1,sizeY1,f])
      ret
  end
## X MAP = 3 GNX; 3 PARAMETERS; 2D, COORD: TRUE;
  defk map3_3para_coord_2D_kernel(d_array1, d_array2, d_array3, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      f(d_array1+id, d_array2+id, d_array3+id, par1, par2, par3, x, y)
    end
  end
  def map3_3para_coord_2D(d_array1, d_array2, d_array3, par1, par2, par3, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeX2 != sizeX3 or sizeY != sizeY2 or sizeY2 != sizeY3 or step != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)

    PolyHok.spawn(&Ske.map3_3para_coord_2D_kernel/10,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,par1,par2,par3,step,sizeX,sizeY,f])
    d_array1
  end
## X MAP = 3 GNX; 3 PARAMETERS; 2D, COORD: TRUE, RETURN: TRUE;
  defk map3_3para_coord_2D_resp_kernel(d_array1, d_array2, d_array3, ret, par1, par2, par3, step, sizeX, sizeY, f) do
    idX = blockIdx.x * blockDim.x + threadIdx.x
    idY = blockIdx.y * blockDim.y + threadIdx.y
    stride = idX + idY * blockDim.x * gridDim.x

    ## Aqui tenho que verificar o 'step'
    if(stride < (sizeX*sizeY)) do
      x = (stride - sizeY * (stride / sizeY))
      y = stride/sizeY
      id = stride*step

      ret[id] = f(d_array1[id], d_array2[id], d_array3[id], par1, par2, par3, x, y)
    end
  end
  def map3_3para_coord_2D_resp(d_array1, d_array2, d_array3, par1, par2, par3, f) do
    {sizeX,sizeY,step} =  case PolyHok.get_shape_gnx(d_array1) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX2,sizeY2,step2} =  case PolyHok.get_shape_gnx(d_array2) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    {sizeX3,sizeY3,step3} =  case PolyHok.get_shape_gnx(d_array3) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end
    if(sizeX != sizeX2 or sizeX2 != sizeX3 or sizeY != sizeY2 or sizeY2 != sizeY3 or step != step2 or step2 != step3) do
      raise "All matrices shall have same shape."
    end

    block_size = 16
    #block_size = 128
    grid_rows = trunc ((sizeX + block_size - 1) / block_size)
    grid_cols = trunc ((sizeY + block_size - 1) / block_size)
    ret = PolyHok.new_gnx(PolyHok.get_shape(d_array1),PolyHok.get_type(d_array1))

    PolyHok.spawn(&Ske.map3_3para_coord_2D_resp_kernel/11,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array1,d_array2,d_array3,ret,par1,par2,par3,step,sizeX,sizeY,f])
    ret
  end

end
