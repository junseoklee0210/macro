# Code for Two-skills Model
# Last Update Apr 17, 2021


function TV(VE::Array{Float64,3}, VN::Array{Float64,3},
        wu::Float64, ws::Float64, r::Float64, β::Float64,
        na::Int64, nz::Int64, Bu::Float64, Bs::Float64, amax::Float64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2};
        compute_policy=false)
    """
    Update Each Value Function by Optimzation
    Notation: E (Employment), N (Not Working)
    """
    TVE = zeros(ne,na,nz)
    TVN = zeros(ne,na,nz)
    
    if compute_policy
        AS = zeros(ne,na,nz)
        HR = zeros(ne,na,nz)
        CS = zeros(ne,na,nz)
    end

    for ie in 1:ne

        if ie == 1
            w = wu  # unskilled
            B = Bu * hbar ^ (1+1/γ) / (1+1/γ)
        else
            w = ws # skilled
            B = Bs * hbar ^ (1+1/γ) / (1+1/γ)
        end
        
        VE′s = [LinearInterpolation(agrid, VE[ie,:,iz]) for iz in 1:nz]
        VN′s = [LinearInterpolation(agrid, VN[ie,:,iz]) for iz in 1:nz]

        for (iz, ez) in enumerate(ezgrid)
            
            objE = (a′ -> log(w * ez * hbar + (1 + r) * a - a′ + eps()) - B
                + β * (sum([max(VE′(a′), VN′(a′)) for (VE′, VN′) in zip(VE′s, VN′s)], weights(Π[iz,:]))) for a in agrid)
            objN = (a′ -> log((1 + r) * a - a′+ eps())
                + β * (sum([max(VE′(a′), VN′(a′)) for (VE′, VN′) in zip(VE′s, VN′s)], weights(Π[iz,:]))) for a in agrid)

            resultsE = maximize.(objE, amin, min.((w * ez * hbar) .+ (1 + r) .* agrid, amax))
            resultsN = maximize.(objN, amin, min.((1 + r) .* agrid, amax))

            TVE[ie,:,iz] = Optim.maximum.(resultsE)
            TVN[ie,:,iz] = Optim.maximum.(resultsN)

            if compute_policy

                for (ia, a) in enumerate(agrid)

                    if TVE[ie,ia,iz] >= TVN[ie,ia,iz]
                        AS[ie,ia,iz] = Optim.maximizer(resultsE[ia])
                        HR[ie,ia,iz] = hbar
                        CS[ie,ia,iz] = w * ez * hbar + (1 + r) * a - AS[ie,ia,iz]
                    else
                        AS[ie,ia,iz] = Optim.maximizer(resultsN[ia])
                        HR[ie,ia,iz] = 0.0
                        CS[ie,ia,iz] = (1 + r) * a - AS[ie,ia,iz]
                    end
                end
            end
        end
    end
    if compute_policy
        return TVE, TVN, AS, HR, CS
    end
    return TVE, TVN
end


function VFI(init_VE::Array{Float64,3}, init_VN::Array{Float64,3},
        wu::Float64, ws::Float64, r::Float64, β::Float64,
        na::Int64, nz::Int64, Bu::Float64, Bs::Float64, amax::Float64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2}, TV;
        max_iter = 1e10, tol = 1e-5)
    """
    Value Function Iteration
    """
    println("VFI")
    VE = init_VE
    VN = init_VN
    for iter in 0:max_iter
        TVE, TVN = TV(VE, VN, wu, ws, r, β, na, nz, Bu, Bs, amax, agrid, ezgrid, Π)
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


function Tμ(μ::Array{Float64,3}, AS::Array{Float64,3}, na::Int64, nz::Int64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2})
    """
    Update Each Discretized PDF(μ)
    """
    μ′ = zeros(ne,na,nz)
    for ie in 1:ne
        for iz in 1:nz
            for ia in 1:na
                ia′h = findfirst(a′-> a′ > AS[ie,ia,iz], agrid)
                if ia′h == nothing
                    μ′[ie,end,:] .+= μ[ie,ia,iz] .* Π[iz,:]
                else
                    ia′l = ia′h - 1
                    wgt = (agrid[ia′h]-AS[ie,ia,iz]) / (agrid[ia′h]-agrid[ia′l])
                    μ′[ie,ia′l,:] .+= μ[ie,ia,iz] .* Π[iz,:] * wgt
                    μ′[ie,ia′h,:] .+= μ[ie,ia,iz] .* Π[iz,:] * (1-wgt)
                end
            end
        end
    end
    return μ′
end;


function μFI(init_μ::Array{Float64,3}, AS::Array{Float64,3}, na::Int64, nz::Int64,
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


function findLK(init_VE::Array{Float64,3}, init_VN::Array{Float64,3},
        init_μ::Array{Float64,3}, Lu::Float64, Ls::Float64, K::Float64,
        β::Float64, s::Float64, na::Int64, nz::Int64,
        Bu::Float64, Bs::Float64, amax::Float64,
        agrid::Array{Float64,1}, ezgrid::Array{Float64,1}, Π::Array{Float64,2},
        TV, VFI, Tμ, μFI)
    """
    Find the Aggregate Labor and Capital using a Shooting Algorithm 
    """
    println("finkLK")
    L = (s * Lu ^ ((θ-1)/θ) + (1-s) * Ls ^ ((θ-1)/θ)) ^ (θ/(θ-1))
    wu = α * (K/L) ^ (1-α) * s * (L/Lu) ^ (1/θ)
    ws = α * (K/L) ^ (1-α) * (1-s) * (L/Ls) ^ (1/θ)
    rmax = 1 / β - 1 - eps()
    rmin = - δ + eps()
    r = min(max((1 - α) * (L / K) ^ α - δ, rmin), rmax)
    
    VE, VN = VFI(init_VE, init_VN, wu, ws, r, β, na, nz,
        Bu, Bs, amax, agrid, ezgrid, Π, TV)
    VE, VN, AS, HR, CS = TV(VE, VN, wu, ws, r, β, na, nz,
        Bu, Bs, amax, agrid, ezgrid, Π; compute_policy=true)
    μ = μFI(init_μ, AS, na, nz, agrid, ezgrid, Π, Tμ)
    
    Lu′ = sum([ez * sum(μ[1,:,iz] .* HR[1,:,iz]) for (iz, ez) in enumerate(ezgrid)])
    Ls′ = sum([ez * sum(μ[2,:,iz] .* HR[2,:,iz]) for (iz, ez) in enumerate(ezgrid)])
    K′ = sum([sum(μ[:,ia,:] .* a) for (ia, a) in enumerate(agrid)])
    
    L′ = (s * Lu′ ^ ((θ-1)/θ) + (1-s) * Ls′ ^ ((θ-1)/θ)) ^ (θ/(θ-1))
    Kd = fzero(K′ -> (1 - α) * (L′/K′) ^ α - r - δ, 1)
    println(K′)
    println(Kd)
    return VE, VN, AS, HR, CS, μ, Lu′, Ls′, K′, Kd
end;


function SteadyState(init_VE::Array{Float64,3}, init_VN::Array{Float64,3},
        init_μ::Array{Float64,3}, init_Lu::Float64, init_Ls::Float64, init_K::Float64,
        β::Float64, s::Float64, na::Int64, nz::Int64, Bu::Float64, Bs::Float64, amax::Float64,
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
    Lu = init_Lu
    Ls = init_Ls
    K = init_K
    damp = convex_combi
    diff = Inf
    
    for iter in 1:max_iter
        println("Lu = $(round(Lu, digits=16))")
        println("Ls = $(round(Ls, digits=16))")
        println("K  = $(round(K, digits=16))")
        println("------------------------------------------------------")
        VE, VN, AS, HR, CS, μ, Lu′, Ls′, K′, Kd = findLK(
            VE, VN, μ, Lu, Ls, K, β, s, na, nz, Bu, Bs,
            amax, agrid, ezgrid, Π, TV, VFI, Tμ, μFI)
        
        diff′ = abs(Lu′ - Lu) / Lu + abs(Ls′ - Ls) / Ls + abs(K′ - K) / K + abs(Kd - K) / K
        
        if diff′ < tol
            return VE, VN, AS, HR, CS, μ, Lu′, Ls′, K′
        else
            # Adjust a dampening factor
            if diff′ > diff
                damp = max(damp * shrink_factor, damp_min)
            else
                damp = min(damp * expand_factor, 0.97)
            end
            
            Lu = Lu * (1.0 - damp) + Lu′ * damp
            Ls = Ls * (1.0 - damp) + Ls′ * damp
            K = K * (1.0 - damp) + (K′+ Kd) / 2 * damp
            
            L = (s * Lu ^ ((θ-1)/θ) + (1-s) * Ls ^ ((θ-1)/θ)) ^ (θ/(θ-1))
            wu = α * (K/L) ^ (1-α) * s * (L/Lu) ^ (1/θ)
            ws = α * (K/L) ^ (1-α) * (1-s) * (L/Ls) ^ (1/θ)

            empu = sum(μ[1,:,:] .* (HR[1,:,:] .> 0)) / sum(μ[1,:,:])
            emps = sum(μ[2,:,:] .* (HR[2,:,:] .> 0)) / sum(μ[2,:,:])
            wdis = wu / ws
            rate = (1 - α) * (L / K) ^ α - δ
            
            println("------------------------------------------------------")
            println("dampening factor: $(round(damp, digits=8))")
            println("current difference: $(round(diff′, digits=8))")
            println("current empu: $(round(empu, digits=8))")
            println("current emps: $(round(emps, digits=8))")
            println("current wdis: $(round(wdis, digits=8))")
            println("current rate: $(round(rate*100, digits=8))")
            println("------------------------------------------------------")
            
        end
        diff = diff′
    end
end;