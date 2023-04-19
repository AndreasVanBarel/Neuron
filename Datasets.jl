module Datasets 

export Sample 
export Dataset

import Base.getproperty
import Base: getindex, setindex!, size

# A single input-output pair (x,y)
struct Sample 
    x
    y
end

struct Dataset 
    samples::Vector{Sample}
end

function getproperty(d::Dataset, sym::Symbol)
    if sym === :x || sym === :inputs
        return [sample.x for sample in d.samples]
    elseif sym === :y || sym === :outputs
        return [sample.y for sample in d.samples]
    else
        return getfield(d,sym)
    end
end

getindex(d::Dataset, key...) = getindex(d.samples, key...)
setindex!(d::Dataset, key...) = setindex(d.samples, key...)
size(d::Dataset) = sizes(d.samples)

end