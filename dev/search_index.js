var documenterSearchIndex = {"docs":
[{"location":"aux/#Auxiliary-Functions-and-Utilities","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"","category":"section"},{"location":"aux/#linalg","page":"Auxiliary Functions and Utilities","title":"Linear Algebra","text":"","category":"section"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"It is essential to be able solve for the updated boldsymbolbeta when we update the boldsymbolomega.  In a typical setting, this corresponds to solving","category":"page"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"(S^astS + N lambda I)boldsymbolbeta = S^ast boldsymboly","category":"page"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"We have included two naive solvers for this problem:","category":"page"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"solve_normal!\nsolve_normal_svd!","category":"page"},{"location":"aux/#ARFF.solve_normal!","page":"Auxiliary Functions and Utilities","title":"ARFF.solve_normal!","text":"solve_normal!(β, S, y_data; λ = 1e-8)\n\nSolve the regularized linear system using the normal equations.\n\nFields\n\nβ - The vector of coefficients that will be obtained\nS - The design matrix\ny_data - y coordinates\nλ = 1e-8 - Regularization parameter\n\n\n\n\n\n","category":"function"},{"location":"aux/#ARFF.solve_normal_svd!","page":"Auxiliary Functions and Utilities","title":"ARFF.solve_normal_svd!","text":"solve_normal_svd!(β, S, y_data; λ = 1e-8)\n\nSolve the regularized linear system using the SVD.\n\nFields\n\nβ - The vector of coefficients that will be obtained\nS - The design matrix\ny_data - y coordinates\nλ = 1e-8 - Regularization parameter\n\n\n\n\n\n","category":"function"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"Other formulations may be more appropriate.  Indeed, in [2], the authors use the regularized loss function in the spirit of Sobolev:","category":"page"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"Sboldsymbolbeta-boldsymboly_2^2 + lambda sum_k (1+omega_k^2)beta_k^2","category":"page"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"We thus require that the user provided linear solver take the form:","category":"page"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"function linear_solver!(β, S, y, ω)\n    # solve for β coefficients\n    β\nend","category":"page"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"Obviously, if one does not need ω for your formulation, as is the case in the original regularization of [1], this argument is just ignored.  For solve_normal! we would implement this as:","category":"page"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"linear_solver! = (β, S, y, ω) -> solve_normal!(β, S, y)","category":"page"},{"location":"aux/#loss","page":"Auxiliary Functions and Utilities","title":"Loss Functions","text":"","category":"section"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"ARFF.mse_loss","category":"page"},{"location":"aux/#ARFF.mse_loss","page":"Auxiliary Functions and Utilities","title":"ARFF.mse_loss","text":"mse_loss(F, data_x, data_y)\n\nMean squared error loss function\n\nFields\n\nF - A FourierModel structure\ndata_x - the x coordinates in training data\ndata_y - the y coordinates in training data\n\n\n\n\n\n","category":"function"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"As the entire framework is built around the mean square loss function, we have included it for convenience.  Other loss functions can be implemented, but they should have the calling sequence:","category":"page"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"function loss_function(F, data_x, data_y)\n    # compute loss \n    return loss\nend","category":"page"},{"location":"aux/#Other-Utilities","page":"Auxiliary Functions and Utilities","title":"Other Utilities","text":"","category":"section"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"optimal_γ","category":"page"},{"location":"aux/#ARFF.optimal_γ","page":"Auxiliary Functions and Utilities","title":"ARFF.optimal_γ","text":"optimal_γ(d::Integer)\n\nCompute the optimal γ parameter as a function of dimension d\n\nFields\n\nd - the dimension of the x coordinate\n\n\n\n\n\n","category":"function"},{"location":"aux/","page":"Auxiliary Functions and Utilities","title":"Auxiliary Functions and Utilities","text":"Following Remark 1 in [1], the optimal gamma corresponds to gamma = 3d -2, which is encoded in the above function.","category":"page"},{"location":"structs/#Data-Structures","page":"Structures","title":"Data Structures","text":"","category":"section"},{"location":"structs/","page":"Structures","title":"Structures","text":"Pages = [\"structs.md\"]","category":"page"},{"location":"structs/#Data-Sets","page":"Structures","title":"Data Sets","text":"","category":"section"},{"location":"structs/","page":"Structures","title":"Structures","text":"Training (and testing data) are stored in DataSet structure.","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"    DataSet","category":"page"},{"location":"structs/#ARFF.DataSet","page":"Structures","title":"ARFF.DataSet","text":"DataSet{TB<:Complex,TW<:AbstractArray{AbstractFloat}}\n\nTraining data containing (x,y) data pairs stored in arrays of x values and arrays of y values.\n\nFields\n\nx - Array of real valued vectors \ny - Array of complex scalars\n\n\n\n\n\n","category":"type"},{"location":"structs/","page":"Structures","title":"Structures","text":"A training set can be constructed as follows","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"using ARFF\nusing Random \nRandom.seed!(100); # for reproducibility\n\nn = 10; # number of sample points\nx = [rand(2) for _ in 1:n];\nf(x) = exp(-x[1]*x[2]); # arbitrary function\ny = f.(x);\n\ndata = DataSet(x,y);\nprintln(data.x)\nprintln(data.y)","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"Additionally, it is often helpful to scale the training data (setting means to zero and variances to unity).  These can be accomplished with the DataScalings structure and the associated functions.","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"    DataScalings\n    get_scalings\n    scale_data!\n    rescale_data!","category":"page"},{"location":"structs/#ARFF.DataScalings","page":"Structures","title":"ARFF.DataScalings","text":"DataScalings{TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}\n\nData structure holding the scalings of a DataSet type.  Used for centering and scaling the data to improve training.\n\nFields\n\nμx - Mean in x\nσ2x - Variance in x\nμy - Mean in y\nσ2y - Variance in y\n\n\n\n\n\n","category":"type"},{"location":"structs/#ARFF.get_scalings","page":"Structures","title":"ARFF.get_scalings","text":"get_scalings(data::DataSet{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}\n\nFind the means and variances of the data for scaling\n\nFields\n\ndata - The training data set\n\n\n\n\n\n","category":"function"},{"location":"structs/#ARFF.scale_data!","page":"Structures","title":"ARFF.scale_data!","text":"scale_data!(data::DataSet{TB,TR,TW}, scalings::DataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}\n\nScale the data set (in-place) according to the specified scalings\n\nFields\n\ndata - Data set to be scale\nscalings - Scalings to apply to data\n\n\n\n\n\n","category":"function"},{"location":"structs/#ARFF.rescale_data!","page":"Structures","title":"ARFF.rescale_data!","text":"rescale_data!(data::DataSet{TB,TR,TW}, scalings::DataScalings{TB,TR,TW}) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR}}\n\nRescale the data set (in-place) according back to the original units\n\nFields\n\ndata - Data set to be scale\nscalings - Scalings to apply to data\n\n\n\n\n\n","category":"function"},{"location":"structs/","page":"Structures","title":"Structures","text":"For example, the following code obtains the means and variances in the data:","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"scalings = get_scalings(data);\nprintln(scalings.μx);\nprintln(scalings.σ2x);\nprintln(scalings.μy);\nprintln(scalings.σ2y);","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"Next, we can scale our data:","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"scale_data!(data, scalings);\n\nusing Statistics\nprintln(mean(data.x));\nprintln(var(data.x));\nprintln(mean(data.y));\nprintln(var(data.y));","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"If needed, we can then undo the scaling,","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"rescale_data!(data, scalings)\n\nprintln(mean(data.x));\nprintln(var(data.x));\nprintln(mean(data.y));\nprintln(var(data.y));","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"Note that these are in agreement with what was contained in our scalings structure.","category":"page"},{"location":"structs/#Fourier-Feature-Models","page":"Structures","title":"Fourier Feature Models","text":"","category":"section"},{"location":"structs/","page":"Structures","title":"Structures","text":"Fourier features approximates functions with K features according to","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"f_rm true(x) approx f(x) = sum_k=1^K beta_k e^i omega_k cdot x","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"In the above expression, xomega_kin mathbbR^d, while beta_kinmathbbC.  Consequently, our model is uniquely determined by the coefficients and the wavenumbers.  ","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"A Fourier features model can be instantiated with a FourierModel structure:","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"    FourierModel","category":"page"},{"location":"structs/#ARFF.FourierModel","page":"Structures","title":"ARFF.FourierModel","text":"FourierModel{TB<:Complex,TW<:AbstractArray{AbstractFloat}}\n\nStructure containing a scalar valued fourier model which will be learned\n\nFields\n\nβ - Array of complex coefficients\nω - Array of wave vectors\n\n\n\n\n\n","category":"type"},{"location":"structs/","page":"Structures","title":"Structures","text":"As an example,","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"Random.seed!(200); # for reproducibility\n\nK = 10;\nd = 2;\n\nF = FourierModel([randn(ComplexF64) for _ in 1:K],  \n    [randn(d) for _ in 1:K]);","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"This defines F with random wavenumbers and amplitudes.  Strictly speaking, this is not required, but it is often helpful within the learning context that we will apply this method.  Function evaluation has been overloaded for a FourierModel, allowing us to evaluate it at points:","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"x_test = [0., 1.];\nprintln(F(x_test));","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"and it can perform a vectorized evaluation:","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"F.(data.x);","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"As is sometimes the case, we may scale our data, train in the scale coordinate, and wish to evaluate new poitns in the original, unscaled, coordinate system. We can accomplish that by passing a DataScalings argument in for evaluation:","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"F(x_test, scalings)","category":"page"},{"location":"structs/#Training-Options","page":"Structures","title":"Training Options","text":"","category":"section"},{"location":"structs/","page":"Structures","title":"Structures","text":"To train a FourierModel, the training options must first be specified by instantiating an ARFFOptions struct with the desired parameters.","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"    ARFFOptions","category":"page"},{"location":"structs/#ARFF.ARFFOptions","page":"Structures","title":"ARFF.ARFFOptions","text":"ARFFOptions{TI<:Integer,TF<:AbstractFloat,TB<:Bool,TL,TS}\n\nData structure containing the training options and parameters\n\nFields\n\nn_epochs - Number of training epochs\nn_ω_steps - Number of internal RWM steps\nδ - RWM proposal step size\nn_burn - Number of epochs before the covariance adaptation begins\nγ - Metropolis-Hastings exponent\nω_max - Maximum wave number norm cutoff\nadapt_covariance - Boolean for adaptivity\nlinear_solve! - User specified solver for the normal equations\nloss - User specified loss function\n\n\n\n\n\n","category":"type"},{"location":"structs/","page":"Structures","title":"Structures","text":"The above arguments are passed in as follows:","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"δ = 1.; # rwm step size\nn_epochs = 10^3; \nn_ω_steps = 10; \nn_burn = 100;\nγ = 1;\nω_max =Inf;\nadapt_covariance = true;\n\nβ_solver! = (β, S, y, ω)-> solve_normal!(β, S, y, λ);\n\nlinear_solve! = (β, S, y, ω)-> reg_β_solver!(β, S, y, λ);\n\nopts = ARFFOptions(n_epochs, n_ω_steps, δ, n_burn, γ, ω_max, adapt_covariance, \n    linear_solve!, ARFF.mse_loss);","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"The linear solver is typed as:","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"function linear_solve!(β, S, y, ω)\n    # in place solver code...\nend","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"which is to say, it can depend on the design matrix S, the response variables, y, and, potentially, the current set of wave numbers, ω.  See Linear Algebra for addititional details.","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"The loss function is to be typed as:","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"function loss(F, x, y)\n    # compute and return loss function...\nend","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"See Loss for addititional details.","category":"page"},{"location":"structs/","page":"Structures","title":"Structures","text":"The γ parameter can be set using optimal_γ, which, in a certain regime, is the optimal value for given dimension, d when xinmathbbR^d.","category":"page"},{"location":"train/#Training","page":"Training","title":"Training","text":"","category":"section"},{"location":"train/","page":"Training","title":"Training","text":"The key function is train_rwm!, which performs in place training on the model. This is implemented to handle the training data in several ways:","category":"page"},{"location":"train/","page":"Training","title":"Training","text":"A single DataSet can be provided and used in every training epoch.\nAn array of DataSet types can be provided, and they will be cycled through each epoch;\nA single DataSet and a minibatch size can be provided, and minibatchs will be generated at each epoch.","category":"page"},{"location":"train/#In-Place-Training","page":"Training","title":"In Place Training","text":"","category":"section"},{"location":"train/","page":"Training","title":"Training","text":"    train_rwm!","category":"page"},{"location":"train/#ARFF.train_rwm!","page":"Training","title":"ARFF.train_rwm!","text":"train_rwm!(F::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}\n\nTrain the Fourier feature model using a random walk Metropolis exploration strategy\n\nFields\n\nF - The FourierModel to be trained\ndata- The DataSet training data\nΣ - Initial covariance matrix for RWM proposals\noptions - ARFFOptions structure specifcying the number epochs, proposal step size, etc.\nshow_progress=true - Display training progress using ProgressMeter\nrecord_loss=true - Evaluate the specified loss function at each epoch and record\n\n\n\n\n\ntrain_rwm!(F::FourierModel{TB,TR,TW}, batched_data::Vector{DataSet{TB,TR,TW}}, Σ::TM, options::ARFFOptions) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}\n\nTrain the Fourier feature model using a random walk Metropolis exploration strategy with batched data, which is cycled through from epoch to epoch.\n\nFields\n\nF - The FourierModel to be trained\nbatched_data- A vector of DataSet training data sets, for the purpose of minibatching.  These are presumed to all be the same size.\nΣ - Initial covariance matrix for RWM proposals\noptions - ARFFOptions structure specifcying the number epochs, proposal step size, etc.\nshow_progress=true - Display training progress using ProgressMeter\nrecord_loss=true - Evaluate the specified loss function at each epoch and record\n\n\n\n\n\ntrain_rwm!(F::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, batch_size::TI, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix,TI<:Integer}\n\nTrain the Fourier feature model using a random walk Metropolis exploration strategy with minibatching, randomly subsampling at each epoch.\n\nF - The FourierModel to be trained\nbatch_size- Minibatch size\nΣ - Initial covariance matrix for RWM proposals\noptions - ARFFOptions structure specifcying the number epochs, proposal step size, etc.\nshow_progress=true - Display training progress using ProgressMeter\nrecord_loss=true - Evaluate the specified loss function at each epoch and record\n\n\n\n\n\n","category":"function"},{"location":"train/","page":"Training","title":"Training","text":"Having created an initial F we can then call","category":"page"},{"location":"train/","page":"Training","title":"Training","text":"Σ_mean, acceptance_rate, loss = train_rwm!(F, data, Σ0, opts);","category":"page"},{"location":"train/","page":"Training","title":"Training","text":"The returned quantities are the mean adapted covariance matrix Σ_mean.  The acceptance_rate is the mean acceptance rate at each epoch, averaged overa the internal steps,  K * n_ω_steps.  The loss is the recorded training loss, stored in opts.loss, at each epoch.","category":"page"},{"location":"train/#Recording-the-Training-Trajectory","page":"Training","title":"Recording the Training Trajectory","text":"","category":"section"},{"location":"train/","page":"Training","title":"Training","text":"We have also included routines which record the model at each epoch.  These are called in the same way as above:","category":"page"},{"location":"train/","page":"Training","title":"Training","text":"F_trajectory, Σ_mean, acceptance_rate, loss = train_rwm(F0, data, Σ0, opts);","category":"page"},{"location":"train/","page":"Training","title":"Training","text":"The functions are analogously named:","category":"page"},{"location":"train/","page":"Training","title":"Training","text":"train_rwm","category":"page"},{"location":"train/#ARFF.train_rwm","page":"Training","title":"ARFF.train_rwm","text":"train_rwm(F₀::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}\n\nTrain the Fourier feature model using a random walk Metropolis exploration strategy. Returns the entire trajectory of models during training.\n\nFields\n\nF₀ - The initial state of the FourierModel to be trained\ndata- The DataSet training data\nΣ - Initial covariance matrix for RWM proposals\noptions - ARFFOptions structure specifcying the number epochs, proposal step size, etc.\nshow_progress=true - Display training progress using ProgressMeter\nrecord_loss=true - Evaluate the specified loss function at each epoch and record\n\n\n\n\n\ntrain_rwm(F₀::FourierModel{TB,TR,TW}, batched_data::Vector{DataSet{TB,TR,TW}}, Σ::TM, options::ARFFOptions; show_progress=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix}\n\nTrain the Fourier feature model using a random walk Metropolis exploration strategy with batched data, which is cycled through from epoch to epoch. Returns the entire trajectory of models during training.\n\nFields\n\nF₀ - The initial state of the FourierModel to be trained\nbatched_data- A vector of DataSet training data sets, for the purpose of minibatching.  These are presumed to all be the same size.\nΣ - Initial covariance matrix for RWM proposals\noptions - ARFFOptions structure specifcying the number epochs, proposal step size, etc.\nshow_progress=true - Display training progress using ProgressMeter\nrecord_loss=true - Evaluate the specified loss function at each epoch and record\n\n\n\n\n\ntrain_rwm(F₀::FourierModel{TB,TR,TW}, data::DataSet{TB,TR,TW}, batch_size::TI, Σ::TM, options::ARFFOptions; show_progress=true, record_loss=true) where {TB<:Complex,TR<:AbstractFloat,TW<:AbstractArray{TR},TM<:AbstractMatrix,TI<:Integer}\n\nTrain the Fourier feature model using a random walk Metropolis exploration strategy with minibatching, randomly subsampling at each epoch. Returns the entire trajectory of models during training.\n\nF₀ - The initial state of the FourierModel to be trained\nbatch_size- Minibatch size\nΣ - Initial covariance matrix for RWM proposals\noptions - ARFFOptions structure specifcying the number epochs, proposal step size, etc.\nshow_progress=true - Display training progress using ProgressMeter\nrecord_loss=true - Evaluate the specified loss function at each epoch and record\n\n\n\n\n\n","category":"function"},{"location":"examples/example1/#Scalar-Example","page":"Scalar Example","title":"Scalar Example","text":"","category":"section"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"This example is taken from Section 5 of [1]. Consider trying to learn the scalar mapping","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"f(x) = mathrmSileft(fracxaright)e^-fracx^22 quad a0","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"where mathrmSi is the Sine integral.  To make the problem somewhat challenging, we take a=10^-3.","category":"page"},{"location":"examples/example1/#Generate-Training-Data","page":"Scalar Example","title":"Generate Training Data","text":"","category":"section"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"First, we will generate and visualize the training data:","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"using SpecialFunctions\nusing Random\nusing Plots\nusing LinearAlgebra\nusing ARFF\n\na = 1e-3;\nf(x) = sinint(x/a) * exp(-0.5 * (x^2));\n\nn_x = 500; # number of training points\nd = 1;\nRandom.seed!(100); # for reproducibility\nx = [0.1*rand(d) for _ in 1:n_x];\ny = [f(x_[1]) for x_ in x];\n\n# store data in DataSet structure\ndata = DataSet(x,y);\n\nscatter([x_[1] for x_ in x], y, label=\"Sample Points\")\nxx = LinRange(0, 0.1, 500);\nplot!(xx, f.(xx), label=\"f(x)\")\nxlabel!(\"x\")","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"Note: When the domain of our target function is mathbbR^1, the x-data must still be stored as an array of arrays of length one, not an array of scalars.  We also do not require DataScalings for this problem.","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"println(x[1:5])","category":"page"},{"location":"examples/example1/#Initialize-Fourier-Model","page":"Scalar Example","title":"Initialize Fourier Model","text":"","category":"section"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"Next, we need to initialize our FourierModel","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"K = 2^6;\nRandom.seed!(200); # for reproducibility\nF0 = FourierModel([1. *randn(ComplexF64) for _ in 1:K],  \n    [randn(d) for _ in 1:K]); nothing","category":"page"},{"location":"examples/example1/#Set-Parameters-and-Trai","page":"Scalar Example","title":"Set Parameters and Trai","text":"","category":"section"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"δ = 10.; # rwm step size\nλ = 1e-8; # regularization\nn_epochs = 10^3; # number of epochs\nn_ω_steps = 10; # number of steps between full β updates\nn_burn = n_epochs ÷ 10; # use 10% of the run for burn in\nγ = optimal_γ(d); \nω_max =Inf; # no cut off\nadapt_covariance = true; \n\nΣ0 = ones(1,1);\n\nβ_solver! = (β, S, y, ω)-> solve_normal!(β, S, y, λ=λ);\n\nopts = ARFFOptions(n_epochs, n_ω_steps, δ, n_burn, γ, ω_max,adapt_covariance, \n    β_solver!, ARFF.mse_loss);\n\nRandom.seed!(1000); # for reproducibility\nF = deepcopy(F0);\nΣ_mean, acceptance_rate, loss= train_rwm!(F, data, Σ0, opts, show_progress=false); nothing ","category":"page"},{"location":"examples/example1/#Evaluate-Results","page":"Scalar Example","title":"Evaluate Results","text":"","category":"section"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"Looking at the trianing loss, we see the model appears to be well trained for the selected width, K:","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"plot(1:length(loss), loss, yscale=:log10, label=\"\")\nxlabel!(\"Epoch\")\nylabel!(\"Loss\")","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"Next, we can verify that we have a high quality approximation of the true f(x):","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"xx = LinRange(0, .1, 500);\nscatter([x_[1] for x_ in x], y, label=\"Sample Points\", legend=:right)\nplot!(xx, f.(xx), label = \"Truth\" )\nplot!(xx, real.([F([x_]) for x_ in xx]),label=\"Learned Model (Real Part)\")\nplot!(xx, imag.([F([x_]) for x_ in xx]),label=\"Learned Model (Imaginary Part)\" )\nxlabel!(\"x\")","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"We can also verify that the training data is well fit:","category":"page"},{"location":"examples/example1/","page":"Scalar Example","title":"Scalar Example","text":"scatter(real.(data.y),real.(F.(data.x)),label=\"Training Data\")\nxx = LinRange(0,2,100);\nplot!(xx, xx, ls=:dash, label=\"\")\nxlabel!(\"Truth\")\nylabel!(\"Prediction\")","category":"page"},{"location":"#ARFF.jl-Documentation","page":"Home","title":"ARFF.jl Documentation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for the adaptive random fourier features (ARFF) package.  This package is built around the methodology presented in [1].","category":"page"},{"location":"","page":"Home","title":"Home","text":"Using the package involves three steps:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Formatting your training data into a DataSet structure\nInitializing a FourierModel structure\nTraining","category":"page"},{"location":"#Overview","page":"Home","title":"Overview","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The essential idea of ARFF is to make an approximation of a true function, f_rm truemathbbR^dto mathbbC, as","category":"page"},{"location":"","page":"Home","title":"Home","text":"f_rm true(x) approx f(x) = sum_k=1^K beta_k e^i omega_k cdot x","category":"page"},{"location":"","page":"Home","title":"Home","text":"where xomega_kin mathbbR^d, while beta_kinmathbbC.  In the naive random Fourier featuer setting, the omega_k are sampled from some known distribution mu, and the beta_k are obtained by classical least squares regression or ridge regression,","category":"page"},{"location":"","page":"Home","title":"Home","text":"(S^astS + N lambda I)boldsymbolbeta = S^ast boldsymboly","category":"page"},{"location":"","page":"Home","title":"Home","text":"where the design matrix, S, is Ntimes K, with entries","category":"page"},{"location":"","page":"Home","title":"Home","text":"S_jk = e^ i omega_k cdot x_j","category":"page"},{"location":"","page":"Home","title":"Home","text":"We presume that we have training data of size N, (x_jy_j)_j=1^N. Other solutions are possible.","category":"page"},{"location":"#Adaptivity","page":"Home","title":"Adaptivity","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"To make the algorithm adaptive, that is to say, to sample the frequencies from an optimal distribution, we use a Random Walk Metropolis scheme described in [1].  The goal is to sample from the variance minimizing distribution, known to be propto hatf(omega).","category":"page"},{"location":"","page":"Home","title":"Home","text":"The strategy is as follows:","category":"page"},{"location":"#Generate-Proposal","page":"Home","title":"Generate Proposal","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Perturb the vector boldsymbolomega of wave numbers with a Gaussian,","category":"page"},{"location":"","page":"Home","title":"Home","text":"boldsymbolomega = boldsymbolomega + delta boldsymbolxi quad boldsymbolxisim N(0 Sigma)","category":"page"},{"location":"","page":"Home","title":"Home","text":"where delta0 is a proposal step size and  Sigma is a covariance matrix. ","category":"page"},{"location":"#Update-Amplitudes","page":"Home","title":"Update Amplitudes","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Compute the proposed amplitudes, boldsymbolbeta for the perturbed wave numbers, by building up the new design matrix and solving the linear system. ","category":"page"},{"location":"#Accept/Reject","page":"Home","title":"Accept/Reject","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Accept/reject each wave vector omega_k with probability","category":"page"},{"location":"","page":"Home","title":"Home","text":"minleft1 fracbeta_k^gammabeta_k^gammaright","category":"page"},{"location":"","page":"Home","title":"Home","text":"where gamma0 is a tuning parameter that plays a role anlogous to inverse temperature.","category":"page"},{"location":"#Training","page":"Home","title":"Training","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Having described a single the RWM step, the core of ARFF training requires  a total number of epochs (n_epoch) and number of RWM steps (n_ω_steps).  The core of the training loop consists of:","category":"page"},{"location":"","page":"Home","title":"Home","text":"for i in 1:n_epochs\n    # solve for β with current ω\n    for j in 1:n_ω_steps\n        # generate an RWM proposal\n            for k in 1:K\n                # accept/reject each ω_k\n            end\n    end\nend","category":"page"},{"location":"#Acknowledgements","page":"Home","title":"Acknowledgements","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Contributors to this project include:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Gideon Simpson \nPetr Plechac\nJerome Troy\nLiam Doherty\nHunter Wages","category":"page"},{"location":"","page":"Home","title":"Home","text":"This work was supported in part by the ARO Cooperative Agreement W911NF2220234.","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"A. Kammonen, J. Kiessling, P. Plecháč, M. Sandberg and A. Szepessy. Adaptive random Fourier features with Metropolis sampling. Foundations of Data Science 2, 309–332 (2020). Accessed on Nov 18, 2023.\n\n\n\nJ. Kiessling, E. Ström and R. Tempone. Wind field reconstruction with adaptive random Fourier features. Proceedings of the Royal Society A: Mathematical, Physical and Engineering Sciences 477, 20210236 (2021). Accessed on May 5, 2023.\n\n\n\nA. Kammonen, J. Kiessling, P. Plecháč, M. Sandberg, A. Szepessy and R. Tempone. Smaller generalization error derived for a deep residual neural network compared with shallow networks. IMA Journal of Numerical Analysis 43, 2585–2632 (2023). Accessed on Nov 12, 2023.\n\n\n\n","category":"page"}]
}
