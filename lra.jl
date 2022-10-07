using LinearAlgebra
using Random
using Printf
using Plots
using Statistics
using Distributions
using SparseArrays
using MatrixMarket

function generate_sampling(n,s,p)
    # n : number of columns
    # s : size of sampling
    # p : probabilities for each column

    # generate an idendity matrix nxn
    # select random s columns from this
    eye = Matrix(1.0I, n,n);
    total = 0;
    if p isa Matrix{Float64}
        for i = 1:n
            eye[i,i] = 1 / sqrt(p[i]*s);
        end
    else
        for i = 1:n
            total += norm(A[:,i])*norm(B[i,:]);
        end
        for i= 1:n
            pni = norm(A[:,i])*norm(B[i,:]) / total;
            eye[i,i] = 1 / sqrt(pni*s)
        end
    end
    rng = MersenneTwister();
    perm = randperm(rng,s);
    S = eye[:,perm];
    return S;   
end

function basicMatrixMulitplication(A, B, c, P)
    # A mxn
    # B nxp
    # c sampling number c <= n
    # P probabilities contains n float. 0 < pi < 1 and sum(pi) = 1;
    _,n = size(A);
    S = generate_sampling(n,c,P);
    A * S , S' * B;
end

# Algorithm 4.1 Randomized Range Finder
# Input A(mxn) matrix, integer l
# Output Q(mxl) matrix, approximates the range of A
function rrf(A, l)
    rng = MersenneTwister()
    _,n = size(A);
    Om = randn(rng, Float64, (n,l))
    Y = A*Om
    Q_,_ = qr(Y)
    Q = Matrix(Q_)
    return Q
end

# Algorithm 4.2 Adaptive Randomized Range Finder
# Input A(mxn) matrix, tolerance eps, integer r as oversampling parameter
# Output Q(mxl) orthonormal with tolerance holds with probability 1-min{m,n}10^-r
function arrf(A,eps,r,plot_step)
    # plot_step is used to generat errors and iterations vectors
    # if plot_step is 0 then no vectors are empty
    # for a positive value of plot_step the value is used for iteration step.
    (m,n) = size(A);
    W = zeros(n,r)
    Y = zeros(m,r)
    Q = zeros(m,1)
    j = 0
    max_err=0
    for i=1:r
        w = randn(n,1)
        W[:,i] = w 
        Y[:,i] = A*w
    end

    for i=1:r
        ny = norm(Y[:,i])
        if ny > eps/(10*sqrt(2/pi))
            max_err = ny
        end
    end
    iteration_step=0;
    iterations = []
    errors = []

    while(max_err > eps)
        iteration_step +=1;
        if plot_step > 0 
            if iteration_step % plot_step == 0
                append!(iterations, iteration_step)
                append!(errors,max_err)
            end
        end
        j += 1
        yj = (1.0I-Q*Q')*Y[:,j]
        qj = yj / norm(yj);
        Y[:,j] = yj;
        if j==1
            Q[:,j] = qj
        else
            Q = cat(Q,qj, dims=2)
        end
        wjr = randn(n,1)
        yjr = (1.0I - Q*Q')*(A*wjr)
        Y = cat(Y,yjr,dims=2)
        Y[:,j+r] = yjr
        for i = j+1:j+r-1
            yi = Y[:,i]
            Y[:,i] = yi - (qj'*yi)*qj
        end
        max_err = 0
        for i= j+1:j+r-1
            ny = norm(Y[:,i])
            if ny > eps/(10*sqrt(2/pi))
                max_err = ny
            end
        end
    end
    return Q,iterations,errors
end    

# Algorithm 4.3 Randomized Power Iteration
# Input A(mxn) matrix, integer l, power q 
# Output Q(mxl) matrix, approximates the range of A
function rpi(A, l, q)
    rng = MersenneTwister()
    _,n = size(A)
    Om = randn(rng, Float64, (n,l))
    Y = (A*A')^q*A*Om
    Q_,_ = qr(Y)
    Q = Matrix(Q_)
    return Q
end

# Algorithm 4.4 Randomized Subspace Iteration
# Input A(mxn) matrix, integer l, power q 
# Output Q(mxl) matrix, approximates the range of A
function rsi(A, l, q)
    rng = MersenneTwister()
    _,n = size(A)
    Om = randn(rng, Float64, (n,l))
    Y = A*Om
    Q = qr(Y)
    for i = 1:q
        @show size(A)
        @show size(Q)
        Y = A*Q
        Q= qr(Y)
    end
    return Q
end

# Algorithm 4.5 Fast Randomized Range Finder
# Input A(mxn) matrix, integer l
# Output Q(mxl) matrix, approximates the range of A
function frrf(A,l)
    m,n = size(A)
    D = rucm(n);
    F = dftg(n);
    R = Matrix(1.0I,n,l)[:,shuffle(1:end)]
    Om = sqrt(n/l)*D*F*R
    Y = A*Om
    q,r = qr(Y)
    Q = Matrix(q)
    return Q
end

# Algorithm 5.1 Direct SVD
# Matrix A(mxn), Q(mxk) matrices with
    # with |A-QQ*A| < epsilon
# Matrices U,S,V, U,V are orthonormal, S nonnegative diagonal matrices

function direct_svd(A,Q)
    B = Q'*A
    Uh,S,V = svd(B)
    U = Q*Uh
    return Q,S,V
end

# Uniform Discrete Fourier Transform Generator
# Input integer n
# Output F(nxn) complex matrix
function dftg(n)
    F = ones(ComplexF64,n,n) 
    for i=1:n
        for j=1:n
        F[i,j] = n^(-0.5)*exp(-2*pi*(i-1)*(j-1)/n)
        end
    end
    return F
end

# Random Complex matrix whose entries are uniformly distributed 
# on unit circle.
# Random n real numbers in [-pi, pi] that is uniformly distributed chosen
# those will be considered as angles of complex numbers on unit circle.
function rucm(n)
    rng = MersenneTwister(1234)
    angles = rand(Uniform(-pi,pi), n,n)
    D = ones(ComplexF64, n,n)
    for i = 1:n
        for j = 1:n
            D[i,j] = exp(angles[i,j]im)
        end
    end
    return D
end

    

# Gram-Schmidt orthornormalization
function gso(A)
    m,n = size(A)
    Q = zeros(m,n)
    v1 = A[:,1]
    Q[:,1] = v1 / norm(v1);
    for k = 2:n
        w = A[:,k]
        v = w
        for j = 1:k-1
            vj = Q[:,j]
            v -= (w'*vj)*vj
        end
        Q[:,k] = v / norm(v)
    end
    return Q;
end

































