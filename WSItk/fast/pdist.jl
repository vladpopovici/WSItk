function (p::Array{Float64, 1}, q::Array{Float64, 1})
    a = (p - q)^2
    b =  p + q
    a[abs(b) < 1e-12] = 0.0
    b[abs(b) < 1e-12] = 1.0
    
    r = a / b
    
    return 0.5*sum(r)
    
    