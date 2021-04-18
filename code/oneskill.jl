# Code for One-skill Model
# Last Update Apr 17, 2021


function TV(VE::Array{Float64,2}, VN::Array{Float64,2},
        w::Float64, r::Float64, β::Float64,
        na::Int64, nz::Int64, B::Float64, amax::Float64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2};
        compute_policy=false)
    """
    Update Each Value Function by Optimzation
    Notation: E (Employment), N (Not Working)
    """
    TVE = zeros(na,nz)
    TVN = zeros(na,nz)
    
    if compute_policy
        AS = zeros(na,nz)
        HR = zeros(na,nz)
        CS = zeros(na,nz)
    end
        
    VE′s = [LinearInterpolation(agrid, VE[:,iz]) for iz in 1:nz]
    VN′s = [LinearInterpolation(agrid, VN[:,iz]) for iz in 1:nz]

    for (iz, ez) in enumerate(ezgrid)

        objE = (a′ -> log(w * ez * hbar + (1 + r) * a - a′ + eps()) - B * hbar ^ (1+1/γ) / (1+1/γ)
            + β * (sum([max(VE′(a′), VN′(a′)) for (VE′, VN′) in zip(VE′s, VN′s)], weights(Π[iz,:]))) for a in agrid)
        objN = (a′ -> log((1 + r) * a - a′+ eps())
            + β * (sum([max(VE′(a′), VN′(a′)) for (VE′, VN′) in zip(VE′s, VN′s)], weights(Π[iz,:]))) for a in agrid)

        resultsE = maximize.(objE, amin, min.((w * ez * hbar) .+ (1 + r) .* agrid, amax))
        resultsN = maximize.(objN, amin, min.((1 + r) .* agrid, amax))

        TVE[:,iz] = Optim.maximum.(resultsE)
        TVN[:,iz] = Optim.maximum.(resultsN)

        if compute_policy

            for (ia, a) in enumerate(agrid)

                if TVE[ia,iz] >= TVN[ia,iz]
                    AS[ia,iz] = Optim.maximizer(resultsE[ia])
                    HR[ia,iz] = hbar
                    CS[ia,iz] = w * ez * hbar + (1 + r) * a - AS[ia,iz]
                else
                    AS[ia,iz] = Optim.maximizer(resultsN[ia])
                    HR[ia,iz] = 0.0
                    CS[ia,iz] = (1 + r) * a - AS[ia,iz]
                end
            end
        end
    end
    
    if compute_policy
        return TVE, TVN, AS, HR, CS
    end
    return TVE, TVN
end


function VFI(init_VE::Array{Float64,2}, init_VN::Array{Float64,2},
        w::Float64, r::Float64, β::Float64,
        na::Int64, nz::Int64, B::Float64, amax::Float64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2}, TV;
        max_iter = 1e10, tol = 1e-5)
    """
    Value Function Iteration
    """
    println("VFI")
    VE = init_VE
    VN = init_VN
    for iter in 0:max_iter
        TVE, TVN = TV(VE, VN, w, r, β, na, nz, B, amax, agrid, ezgrid, Π)
        diff = maximum(abs.(TVE .- VE)) + maximum(abs.(TVN .- VN))
        if diff < tol
            return TVE, TVN
        else
            if iter % 500 == 0
                println(diff)
            end
            VE = TVE
            VN = TVN
        end
    end
    println("Not Converged!")
end;


function Tμ(μ::Array{Float64,2}, AS::Array{Float64,2}, na::Int64, nz::Int64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2})
    """
    Update Each Discretized PDF(μ)
    """
    μ′ = zeros(na,nz)
    for iz in 1:nz
        for ia in 1:na
            ia′h = findfirst(a′-> a′ > AS[ia,iz], agrid)
            if ia′h == nothing
                μ′[end,:] .+= μ[ia,iz] .* Π[iz,:]
            else
                ia′l = ia′h - 1
                wgt = (agrid[ia′h]-AS[ia,iz]) / (agrid[ia′h]-agrid[ia′l])
                μ′[ia′l,:] .+= μ[ia,iz] .* Π[iz,:] * wgt
                μ′[ia′h,:] .+= μ[ia,iz] .* Π[iz,:] * (1-wgt)
            end
        end
    end
    return μ′
end;


function μFI(init_μ::Array{Float64,2}, AS::Array{Float64,2}, na::Int64, nz::Int64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2}, Tμ;
        max_iter=10e10, tol=1e-6)
    """
    PDF(μ) Iteration
    """
    println("μFI")
    μ = init_μ
    for iter in 1:max_iter
        μ′ = Tμ(μ, AS, na, nz, agrid, ezgrid, Π)
        diff = maximum(abs.(μ′ .- μ))
        if diff < tol
            return μ′
        else
            if iter % 1000 == 0
                println(diff)
            end
            μ = μ′
        end
    end
    println("Not Converged!")
end;


function findLK(init_VE::Array{Float64,2}, init_VN::Array{Float64,2},
        init_μ::Array{Float64,2}, L::Float64, K::Float64,
        β::Float64, na::Int64, nz::Int64, B::Float64, amax::Float64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2},
        TV, VFI, Tμ, μFI)
    """
    Find the Aggregate Labor and Captial using a Shooting Algorithm 
    """
    println("finkLK")
    w = α * (K/L) ^ (1-α)
    rmax = 1 / β - 1 - eps()
    rmin = - δ + eps()
    r = min(max((1 - α) * (L / K) ^ α - δ, rmin), rmax)
    
    VE, VN = VFI(init_VE, init_VN, w, r, β, na, nz, B, amax, agrid, ezgrid, Π, TV)
    VE, VN, AS, HR, CS = TV(VE, VN, w, r, β, na, nz, B, amax, agrid, ezgrid, Π; compute_policy=true)
    μ = μFI(init_μ, AS, na, nz, agrid, ezgrid, Π, Tμ)
    
    L′ = sum([ez * sum(μ[:,iz] .* HR[:,iz]) for (iz, ez) in enumerate(ezgrid)])
    K′ = sum([sum(μ[ia,:] .* a) for (ia, a) in enumerate(agrid)])
    Kd = fzero(K′ -> (1 - α) * (L′/K′) ^ α - r - δ, 1)
    println(K′)
    println(Kd)
    return VE, VN, AS, HR, CS, μ, L′, K′, Kd
end;


function SteadyState(init_VE::Array{Float64,2}, init_VN::Array{Float64,2},
        init_μ::Array{Float64,2}, init_L::Float64, init_K::Float64,
        β::Float64, na::Int64, nz::Int64, B::Float64, amax::Float64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2},
        TV, VFI, Tμ, μFI, findLK;
        max_iter=1e10, tol=0.001,
        convex_combi::Float64=0.05, shrink_factor::Float64=0.5,
        expand_factor::Float64=1.1, damp_min::Float64=0.02)
    """
    Find the Steady State using a Shooting Algorithm
    """
    VE = init_VE
    VN = init_VN
    μ = init_μ
    L = init_L
    K = init_K
    damp = convex_combi
    diff = Inf
    
    for iter in 1:max_iter
        println("L = $(round(L, digits=16))")
        println("K = $(round(K, digits=16))")
        println("------------------------------------------------------")
        VE, VN, AS, HR, CS, μ, L′, K′, Kd = findLK(
            VE, VN, μ, L, K, β, na, nz, B,
            amax, agrid, ezgrid, Π, TV, VFI, Tμ, μFI)
        
        diff′ = abs(L′ - L) / L + abs(K′ - K) / K + abs(Kd - K) / K
        
        if diff′ < tol
            return VE, VN, AS, HR, CS, μ, L′, K′
        else
            # Adjust a dampening factor
            if diff′ > diff
                damp = max(damp * shrink_factor, damp_min)
            else
                damp = min(damp * expand_factor, 0.97)
            end
            
            L = L * (1.0 - damp) + L′ * damp
            K = K * (1.0 - damp) + (K′+ Kd) / 2 * damp
            
            emp = sum(μ .* (HR .> 0))
            rate = (1 - α) * (L / K) ^ α - δ
            
            println("------------------------------------------------------")
            println("dampening factor: $(round(damp, digits=8))")
            println("current difference: $(round(diff′, digits=8))")
            println("current emp : $(round(emp, digits=8))")
            println("current rate: $(round(rate*100, digits=8))")
            println("------------------------------------------------------")
            
        end
        diff = diff′
    end
end;