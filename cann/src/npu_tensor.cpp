#include "strata/cann/npu_tensor.hpp"
#include "aclnn_call.hpp"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_sub.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_div.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_copy.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_sigmoid.h>
#include <aclnnop/aclnn_exp.h>
#include <aclnnop/aclnn_log.h>
#include <aclnnop/aclnn_sqrt.h>
#include <aclnnop/aclnn_rsqrt.h>
#include <aclnnop/aclnn_sin.h>
#include <aclnnop/aclnn_cos.h>
#include <aclnnop/aclnn_abs.h>
#include <aclnnop/aclnn_sign.h>
#include <aclnnop/aclnn_relu.h>
#include <aclnnop/aclnn_softplus.h>
#include <aclnnop/aclnn_neg.h>
#include <aclnnop/aclnn_pow.h>
#include <aclnnop/aclnn_clamp.h>
#include <aclnnop/aclnn_reduce_sum.h>
#include <aclnnop/aclnn_mean.h>
#include <aclnnop/aclnn_softmax.h>
#include <aclnnop/aclnn_matmul.h>
#include <aclnnop/aclnn_cat.h>
#include <aclnnop/aclnn_index_select.h>
#include <aclnnop/aclnn_topk.h>
#include <aclnnop/aclnn_logsumexp.h>

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>

namespace strata::cann::npu {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw Error(std::string("npu tensor: ") + message, -1);
}
int64_t count(const std::vector<int64_t>& shape) {
    int64_t result = 1;
    for (int64_t size : shape) {
        require(size > 0, "zero or negative dimensions are unsupported");
        require(result <= std::numeric_limits<int64_t>::max() / size, "shape overflow");
        result *= size;
    }
    return result;
}
std::vector<int64_t> dense_strides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> result(shape.size());
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) { result[i] = stride; stride *= shape[i]; }
    return result;
}
bool dense(const ND& x) {
    int64_t stride = 1;
    for (size_t i = x.shape.size(); i-- > 0;) {
        if (x.shape[i] != 1 && x.strides[i] != stride) return false;
        stride *= x.shape[i];
    }
    return true;
}
int64_t axis(int64_t dim, size_t rank) {
    if (dim < 0) dim += static_cast<int64_t>(rank);
    require(dim >= 0 && dim < static_cast<int64_t>(rank), "axis outside rank");
    return dim;
}
aclDataType dtype(Type type) {
    switch (type) {
        case Type::F32: return ACL_FLOAT;
        case Type::F16: return ACL_FLOAT16;
        case Type::BF16: return ACL_BF16;
        case Type::I64: return ACL_INT64;
        case Type::I32: return ACL_INT32;
        case Type::U8: return ACL_UINT8;
    }
    throw Error("npu tensor: invalid dtype", -1);
}
void validate(const ND& x) {
    require(bool(x.storage), "missing storage");
    require(x.shape.size() == x.strides.size(), "shape/stride rank mismatch");
    require(x.offset >= 0, "negative storage offset");
    count(x.shape);
    int64_t last = x.offset;
    for (size_t i = 0; i < x.shape.size(); ++i) {
        require(x.strides[i] >= 0, "negative stride unsupported");
        require(x.shape[i] == 1 || x.strides[i] <= (std::numeric_limits<int64_t>::max()-last)/(x.shape[i]-1), "stride overflow");
        last += (x.shape[i]-1)*x.strides[i];
    }
    require(static_cast<uint64_t>(last) < x.storage->size()/type_bytes(x.type), "view exceeds storage");
}
struct Handle {
    aclTensor* value;
    int ordinal;
    explicit Handle(const ND& x) : ordinal(x.storage ? x.storage->device_ordinal() : -1) {
        validate(x);
        const int64_t storage_size = static_cast<int64_t>(x.storage->size()/type_bytes(x.type));
        value = aclCreateTensor(x.shape.data(), x.shape.size(), dtype(x.type), x.strides.data(), x.offset,
                                ACL_FORMAT_ND, &storage_size, 1, x.storage->data());
        require(value != nullptr, "aclCreateTensor failed");
    }
    ~Handle() { if(value) { auto* owned=value; defer_device_cleanup(ordinal,[owned]{aclDestroyTensor(owned);}); } }
    Handle(const Handle&) = delete;
};
struct Scalar {
    aclScalar* value;
    int ordinal;
    explicit Scalar(float x, Stream& stream) : value(aclCreateScalar(&x, ACL_FLOAT)), ordinal(stream.device_ordinal()) { require(value != nullptr, "aclCreateScalar failed"); }
    ~Scalar() { auto* owned=value; defer_device_cleanup(ordinal,[owned]{aclDestroyScalar(owned);}); }
};
struct Axes {
    aclIntArray* value;
    int ordinal;
    explicit Axes(int64_t dim, Stream& stream) : value(aclCreateIntArray(&dim, 1)), ordinal(stream.device_ordinal()) { require(value != nullptr, "aclCreateIntArray failed"); }
    ~Axes() { auto* owned=value; defer_device_cleanup(ordinal,[owned]{aclDestroyIntArray(owned);}); }
};
std::vector<int64_t> broadcast(const std::vector<int64_t>& a, const std::vector<int64_t>& b) {
    std::vector<int64_t> out(std::max(a.size(),b.size()),1);
    for (size_t back = 0; back < out.size(); ++back) {
        const int64_t av=back<a.size()?a[a.size()-1-back]:1, bv=back<b.size()?b[b.size()-1-back]:1;
        require(av==bv || av==1 || bv==1, "broadcast shape mismatch");
        out[out.size()-1-back]=std::max(av,bv);
    }
    return out;
}
std::vector<int64_t> reduced(const ND& x, int64_t dim, bool keepdim) {
    auto shape=x.shape;
    if (keepdim) shape[dim]=1; else shape.erase(shape.begin()+dim);
    return shape;
}
}

size_t type_bytes(Type type) {
    switch(type) {
        case Type::F32: case Type::I32: return 4;
        case Type::F16: case Type::BF16: return 2;
        case Type::I64: return 8;
        case Type::U8: return 1;
    }
    throw Error("npu tensor: invalid dtype", -1);
}
int64_t ND::numel() const { return count(shape); }
void* ND::data() const { validate(*this); return static_cast<char*>(storage->data()) + offset*type_bytes(type); }
ND empty(const std::vector<int64_t>& shape, Type type) {
    const auto elements=count(shape);
    require(static_cast<uint64_t>(elements)<=std::numeric_limits<size_t>::max()/type_bytes(type), "allocation size overflow");
    return {std::make_shared<Buffer>(elements*type_bytes(type)),shape,dense_strides(shape),type,0};
}
ND zeros(const std::vector<int64_t>& shape, Type type) {
    ND out=empty(shape,type);
    zero_device(out.data(),out.storage->size(),out.storage->device_ordinal());
    return out;
}
ND from_host(const void* data,const std::vector<int64_t>& shape,Type type) {
    ND out=empty(shape,type); out.storage->copy_from_host(data,out.storage->size()); return out;
}
ND wrap(std::shared_ptr<Buffer> storage,const std::vector<int64_t>& shape,Type type) {
    ND out{std::move(storage),shape,dense_strides(shape),type,0}; validate(out); return out;
}
void copy(Stream& stream,const ND& src,ND& dst) {
    require(src.type==dst.type && src.shape==dst.shape,"copy shape/type mismatch");
    if(src.storage==dst.storage && src.offset==dst.offset && src.strides==dst.strides) return;
    // Overlapping views are copied through a separate device allocation.
    if(src.storage==dst.storage) { ND tmp=empty(src.shape,src.type);copy(stream,src,tmp);copy(stream,tmp,dst);return; }
    Handle a(src),out(dst);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnInplaceCopyGetWorkspaceSize(out.value,a.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnInplaceCopy(ws,size,ex,s);},"npu InplaceCopy",true);
}
ND contiguous(Stream& stream,const ND& x) {
    validate(x);if(dense(x)) return x;
    ND out=empty(x.shape,x.type);copy(stream,x,out);return out;
}
ND reshape(Stream& stream,const ND& x,const std::vector<int64_t>& requested) {
    auto shape=requested;int missing=-1;int64_t known=1;
    for(size_t i=0;i<shape.size();++i) {
        if(shape[i]==-1) {require(missing<0,"multiple inferred dimensions");missing=static_cast<int>(i);}
        else {require(shape[i]>0,"invalid reshape dimension");require(known<=std::numeric_limits<int64_t>::max()/shape[i],"reshape overflow");known*=shape[i];}
    }
    if(missing>=0) {require(x.numel()%known==0,"inferred reshape mismatch");shape[missing]=x.numel()/known;}
    require(count(shape)==x.numel(),"reshape element count mismatch");
    ND out=contiguous(stream,x);out.shape=shape;out.strides=dense_strides(shape);return out;
}
ND permute(const ND& x,const std::vector<int64_t>& dims) {
    require(dims.size()==x.shape.size(),"permutation rank mismatch");ND out=x;
    std::vector<bool> used(dims.size(),false);
    for(size_t i=0;i<dims.size();++i) {const auto d=axis(dims[i],dims.size());require(!used[d],"duplicate permutation axis");used[d]=true;out.shape[i]=x.shape[d];out.strides[i]=x.strides[d];}
    validate(out);return out;
}
ND slice(const ND& x,int64_t dim,int64_t start,int64_t length) {
    dim=axis(dim,x.shape.size());if(start<0)start+=x.shape[dim];
    require(start>=0 && length>0 && start<=x.shape[dim]-length,"slice outside axis");
    ND out=x;out.offset+=start*x.strides[dim];out.shape[dim]=length;validate(out);return out;
}
ND cast(Stream& stream,const ND& x,Type type) {
    if(x.type==type)return x;
    ND out=empty(x.shape,type);Handle a(x),b(out);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnCastGetWorkspaceSize(a.value,dtype(type),b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnCast(ws,size,ex,s);},"npu Cast",true);return out;
}

#define NPU_UNARY_CASE(kind, op) case Unary::kind: \
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnn##op##GetWorkspaceSize(a.value,b.value,ws,ex);}, \
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnn##op(ws,size,ex,s);},"npu " #op,true); break
ND unary(Stream& stream,const ND& x,Unary op) {
    ND out=empty(x.shape,x.type);Handle a(x),b(out);
    switch(op) {
        NPU_UNARY_CASE(Silu,Silu); NPU_UNARY_CASE(Sigmoid,Sigmoid); NPU_UNARY_CASE(Exp,Exp);
        NPU_UNARY_CASE(Log,Log); NPU_UNARY_CASE(Sqrt,Sqrt); NPU_UNARY_CASE(Rsqrt,Rsqrt);
        NPU_UNARY_CASE(Sin,Sin); NPU_UNARY_CASE(Cos,Cos); NPU_UNARY_CASE(Abs,Abs);
        NPU_UNARY_CASE(Sign,Sign); NPU_UNARY_CASE(Relu,Relu); NPU_UNARY_CASE(Neg,Neg);
        case Unary::Softplus: {
            Scalar beta(1,stream),threshold(20,stream);
            execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnSoftplusGetWorkspaceSize(a.value,beta.value,threshold.value,b.value,ws,ex);},
                [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnSoftplus(ws,size,ex,s);},"npu Softplus",true);break;
        }
    }
    return out;
}
#undef NPU_UNARY_CASE

ND binary(Stream& stream,const ND& x,const ND& y,Binary op) {
    require(x.type==y.type,"binary operands need identical dtype");
    ND out=empty(broadcast(x.shape,y.shape),x.type);Handle a(x),b(y),c(out);Scalar alpha(1,stream);
    switch(op) {
        case Binary::Add:
            execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnAddGetWorkspaceSize(a.value,b.value,alpha.value,c.value,ws,ex);},
                [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnAdd(ws,size,ex,s);},"npu Add",true);break;
        case Binary::Sub:
            execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnSubGetWorkspaceSize(a.value,b.value,alpha.value,c.value,ws,ex);},
                [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnSub(ws,size,ex,s);},"npu Sub",true);break;
        case Binary::Mul:
            execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnMulGetWorkspaceSize(a.value,b.value,c.value,ws,ex);},
                [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnMul(ws,size,ex,s);},"npu Mul",true);break;
        case Binary::Div:
            execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnDivGetWorkspaceSize(a.value,b.value,c.value,ws,ex);},
                [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnDiv(ws,size,ex,s);},"npu Div",true);break;
    }
    return out;
}
ND scale(Stream& stream,const ND& x,float scalar) {
    ND out=empty(x.shape,x.type);Handle a(x),b(out);Scalar v(scalar,stream);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnMulsGetWorkspaceSize(a.value,v.value,b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnMuls(ws,size,ex,s);},"npu Muls",true);return out;
}
ND add_scalar(Stream& stream,const ND& x,float scalar) {
    ND out=empty(x.shape,x.type);Handle a(x),b(out);Scalar v(scalar,stream),alpha(1,stream);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnAddsGetWorkspaceSize(a.value,v.value,alpha.value,b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnAdds(ws,size,ex,s);},"npu Adds",true);return out;
}
ND pow_scalar(Stream& stream,const ND& x,float scalar) {
    ND out=empty(x.shape,x.type);Handle a(x),b(out);Scalar v(scalar,stream);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnPowTensorScalarGetWorkspaceSize(a.value,v.value,b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnPowTensorScalar(ws,size,ex,s);},"npu PowTensorScalar",true);return out;
}
ND clamp_min(Stream& stream,const ND& x,float scalar) {
    ND out=empty(x.shape,x.type);Handle a(x),b(out);Scalar v(scalar,stream);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnClampMinGetWorkspaceSize(a.value,v.value,b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnClampMin(ws,size,ex,s);},"npu ClampMin",true);return out;
}
ND sum(Stream& stream,const ND& x,int64_t dim,bool keepdim) {
    dim=axis(dim,x.shape.size());ND out=empty(reduced(x,dim,keepdim),x.type);Handle a(x),b(out);Axes axes(dim,stream);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnReduceSumGetWorkspaceSize(a.value,axes.value,keepdim,dtype(x.type),b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnReduceSum(ws,size,ex,s);},"npu ReduceSum",true);return out;
}
ND mean(Stream& stream,const ND& x,int64_t dim,bool keepdim) {
    dim=axis(dim,x.shape.size());ND out=empty(reduced(x,dim,keepdim),x.type);Handle a(x),b(out);Axes axes(dim,stream);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnMeanGetWorkspaceSize(a.value,axes.value,keepdim,dtype(x.type),b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnMean(ws,size,ex,s);},"npu Mean",true);return out;
}
ND softmax(Stream& stream,const ND& x,int64_t dim) {
    dim=axis(dim,x.shape.size());ND out=empty(x.shape,x.type);Handle a(x),b(out);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnSoftmaxGetWorkspaceSize(a.value,dim,b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnSoftmax(ws,size,ex,s);},"npu Softmax",true);return out;
}
ND matmul(Stream& stream,const ND& x,const ND& y) {
    require(x.shape.size()>=2 && y.shape.size()>=2,"matmul requires matrix dimensions");
    require(x.type==y.type && x.shape.back()==y.shape[y.shape.size()-2],"matmul shape/type mismatch");
    auto shape=broadcast({x.shape.begin(),x.shape.end()-2},{y.shape.begin(),y.shape.end()-2});
    shape.push_back(x.shape[x.shape.size()-2]);shape.push_back(y.shape.back());
    ND out=empty(shape,x.type);Handle a(x),b(y),c(out);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnMatmulGetWorkspaceSize(a.value,b.value,c.value,0,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnMatmul(ws,size,ex,s);},"npu Matmul",true);return out;
}
ND cat(Stream& stream,const std::vector<ND>& xs,int64_t dim) {
    require(!xs.empty(),"cat requires tensors");dim=axis(dim,xs[0].shape.size());auto shape=xs[0].shape;shape[dim]=0;
    std::vector<std::unique_ptr<Handle>> handles;std::vector<const aclTensor*> inputs;
    for(const ND& x:xs) {
        require(x.type==xs[0].type && x.shape.size()==shape.size(),"cat shape/type mismatch");
        for(size_t d=0;d<shape.size();++d)if(d!=static_cast<size_t>(dim))require(x.shape[d]==shape[d],"cat non-axis mismatch");
        shape[dim]+=x.shape[dim];handles.push_back(std::make_unique<Handle>(x));inputs.push_back(handles.back()->value);
    }
    ND out=empty(shape,xs[0].type);Handle result(out);
    aclTensorList* list=aclCreateTensorList(inputs.data(),inputs.size());require(list!=nullptr,"aclCreateTensorList failed");
    // aclCreateTensorList takes ownership of its aclTensor descriptors. The list
    // destroys them; leaving ownership in Handle would double-free descriptors
    // and corrupt later operators. Device storage remains owned by the ND inputs.
    for(auto& handle:handles) handle->value=nullptr;
    try {
        execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnCatGetWorkspaceSize(list,dim,result.value,ws,ex);},
            [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnCat(ws,size,ex,s);},"npu Cat",true);
    } catch(...) {defer_device_cleanup(stream.device_ordinal(),[list]{aclDestroyTensorList(list);});throw;}
    defer_device_cleanup(stream.device_ordinal(),[list]{aclDestroyTensorList(list);});return out;
}
ND index_select(Stream& stream,const ND& x,int64_t dim,const ND& indices) {
    dim=axis(dim,x.shape.size());require(indices.shape.size()==1 && (indices.type==Type::I64 || indices.type==Type::I32),"index_select needs integer vector");
    auto shape=x.shape;shape[dim]=indices.numel();ND out=empty(shape,x.type);Handle a(x),index(indices),b(out);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnIndexSelectGetWorkspaceSize(a.value,dim,index.value,b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnIndexSelect(ws,size,ex,s);},"npu IndexSelect",true);return out;
}
std::pair<ND,ND> topk(Stream& stream,const ND& x,int64_t k,int64_t dim) {
    dim=axis(dim,x.shape.size());require(k>0 && k<=x.shape[dim],"invalid topk size");auto shape=x.shape;shape[dim]=k;
    ND values=empty(shape,x.type),indices=empty(shape,Type::I64);Handle a(x),b(values),c(indices);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnTopkGetWorkspaceSize(a.value,k,dim,true,true,b.value,c.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnTopk(ws,size,ex,s);},"npu Topk",true);return {values,indices};
}
ND logsumexp(Stream& stream,const ND& x,int64_t dim,bool keepdim) {
    dim=axis(dim,x.shape.size());ND out=empty(reduced(x,dim,keepdim),x.type);Handle a(x),b(out);Axes axes(dim,stream);
    execute(stream,[&](uint64_t* ws,aclOpExecutor** ex){return aclnnLogSumExpGetWorkspaceSize(a.value,axes.value,keepdim,b.value,ws,ex);},
        [&](void* ws,uint64_t size,aclOpExecutor* ex,aclrtStream s){return aclnnLogSumExp(ws,size,ex,s);},"npu LogSumExp",true);return out;
}
std::vector<float> read_f32(Stream& stream,const ND& x) {
    ND out=contiguous(stream,cast(stream,x,Type::F32));stream.synchronize();std::vector<float> host(out.numel());
    copy_to_host(out.data(),host.data(),host.size()*sizeof(float));return host;
}
std::vector<int64_t> read_i64(Stream& stream,const ND& x) {
    ND out=contiguous(stream,cast(stream,x,Type::I64));stream.synchronize();std::vector<int64_t> host(out.numel());
    copy_to_host(out.data(),host.data(),host.size()*sizeof(int64_t));return host;
}
}
